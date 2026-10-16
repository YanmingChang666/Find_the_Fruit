# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from collections.abc import Sequence

import omni.usd
import isaacsim.core.utils.stage as stage_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .assets import PLANT_USD, BACKGROUND_USD
from .env_cfg import FindTheFruitEnvCfg, MASK_SIZE
from .scene_builder import add_plant, spawn_plant
from .vision_utils import quaternion_to_rotation_matrix, create_batched_masks, calculate_plant_mask


# 本文件定义强化学习任务本身；PPO 网络和优化过程由 train.py / skrl 负责。
# DirectRLEnv 管理仿真循环，本类实现场景、动作、观测、奖励、终止与重置回调。
# 下文 N 表示并行环境数，K 表示水果数，H/W 表示图像高/宽。
class FindTheFruitEnv(DirectRLEnv):
    cfg: FindTheFruitEnvCfg

    def __init__(self, cfg: FindTheFruitEnvCfg, render_mode: str | None = None, **kwargs):
        # 父类创建仿真上下文、调用本类 _setup_scene，并初始化场景及设备。
        # 因此机器人和相机等对象在下面可以直接访问。
        super().__init__(cfg, render_mode, **kwargs)

        # 将配置中的关节名称映射为机器人内部索引；后续通过这些索引读写关节。
        # find_bodies 返回索引和名称，_arm_pos_idx[0] 是末端连杆的索引集合。
        self._arm_dof_idx, _ = self._robot.find_joints(self.cfg.arm_dof_name)
        self._arm_pos_idx = self._robot.find_bodies(["left_arm_l6"])
        self.action_scale = self.cfg.action_scale

        # 保存机器人状态张量；关节速度虽可读取，但当前没有放入策略观测。
        self.joint_pos = self._robot.data.joint_pos
        self.joint_vel = self._robot.data.joint_vel

        # 保留的图像队列目前未用于观测堆叠，策略实际接收单次图像。
        self.image_sequences = deque(maxlen=20)
        # 将六个初始角度从度转换为弧度，再复制到 N 个环境，形状为 (N, 6)。
        # 重置时还会随机改写第三个受控关节，避免每回合从完全相同姿态开始。
        self.root_init_angles = (
            torch.deg2rad(torch.tensor([0, -90, 0, 120, -120, 180], device=self.device))
            .repeat(self.num_envs, 1)
            .view(-1, len(self.cfg.arm_dof_name))
        )
        self.last_angles = self.root_init_angles.clone()

        # 相机投影矩阵：当前实现仅在初始化时计算一次。
        # K 为内参 (N,3,3)；以下按代码选用的四元数约定构造 R 和 t=-R*C，
        # 拼成 (N,3,4) 矩阵 [R|t]，再计算 P=K[R|t]。
        # 此处沿用现有坐标约定；移植相机或更改姿态时需核对旋转方向和像素翻转。
        # 若相机运动，缓存的投影矩阵不会自动随运动重新计算。
        self.camera_intrinsics = self._tiled_camera.data.intrinsic_matrices
        rotation_matrix = quaternion_to_rotation_matrix(self._tiled_camera.data.quat_w_opengl)
        translation_vector = -torch.bmm(rotation_matrix, self._tiled_camera.data.pos_w.unsqueeze(-1))
        extrinsic_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)
        self.projection_matrix = torch.bmm(self.camera_intrinsics, extrinsic_matrix)

        # 缓存各水果的初始世界质心状态，形状为 (N,K,13)：
        # 位置 3 + 四元数 4 + 线速度 3 + 角速度 3，作为回合重置的基准。
        self.root_cube_positions = torch.stack(
            [cube.data.root_com_state_w.clone() for cube in self.goal_cubes], dim=1
        )
        # 每个环境维护一个目标编号，初始化为 0，重置时再随机选择。
        self.target_cube_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 保存所有候选水果的二值掩码；观测中只取当前目标的那个通道。
        # 现有缓冲区轴顺序写作 (N,W,H,K)，默认 W=H=256；改非方形图像时需检查。
        self.cube_maskas = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, self.cfg.num_goal_cubes),
            device=self.device,
            dtype=torch.float32,
        )
        self.rgb_image = torch.zeros(
            (self.num_envs, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 3),
            device=self.device,
            dtype=torch.float32,
        )
        # 保留的辅助缓存：当前奖励用 detection_sequence 判断连续可见，
        # 并未使用 occlusion_time_below_threshold 这个计时变量。
        self.occlusion_time_below_threshold = torch.zeros(self.num_envs, 1, device=self.device)
        self.joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

    def close(self):
        # 交由父类释放环境、传感器等资源；仿真应用本身由训练/演示入口关闭。
        super().close()

    # ------------------------------------------------------------------
    # 场景搭建：资产创建、并行环境复制、物理连接与传感器注册
    # ------------------------------------------------------------------

    def _setup_scene(self):
        # USD Stage 是场景对象树。机器人使用关节化资产，相机使用批量渲染传感器。
        # env_.* 路径模式用于匹配各个并行环境，/World/ground 为公共地面。
        self.stage = stage_utils.get_current_stage()
        self._robot = Articulation(self.cfg.robot_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # 加载植物 USD 的几何资产；可变形属性与连接约束在后面的 add_plant 设置。
        spawn_plant(
            "/World/envs/env_.*/Plant1",
            translation=(0.06, -0.25, 0.0),
            usd_path=PLANT_USD,
            device=self.device,
        )

        # 每个环境创建球形光源，提供相机 RGB 渲染所需的照明。
        light_cfg = sim_utils.SphereLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), radius=1.0)
        light_cfg.func("/World/envs/env_.*/Light", light_cfg, translation=(0.0, -0.4, 2.0))

        # 背景 USD 与额外的薄长方体用于构建拍摄背景和场景表面。
        bg_color = (0.6, 0.6, 0.6)
        background_cfg = sim_utils.UsdFileCfg(
            usd_path=BACKGROUND_USD, scale=(1.0, 1.0, 1.0)
        )
        background_cfg.func(
            "/World/envs/env_.*/BackgroundPlane",
            background_cfg,
            translation=(0.0, 0.25, 0.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        ground_cfg = sim_utils.CuboidCfg(
            size=(1.25, 0.25, 0.01), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=bg_color)
        )
        ground_cfg.func("/World/envs/env_.*/GroundPlane2", ground_cfg, translation=(0.0, -0.35, 0.0))

        # 用彩色球体表示候选水果，半径 0.03 m；变量名虽为 cube，几何体实际为 Sphere。
        # 默认 K=2，依次为红球、绿球；颜色仅区分候选对象，不表示成功或失败。
        # disable_gravity 只关闭重力，并不等同于把刚体设置成静态或固定约束。
        fruit_colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        self.goal_cubes = []
        for i in range(self.cfg.num_goal_cubes):
            cube_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/GoalCube_{i}",
                spawn=sim_utils.SphereCfg(
                    radius=0.03,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=fruit_colors[i % len(fruit_colors)]),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.4, 0.28)),
            )
            self.goal_cubes.append(RigidObject(cube_cfg))

        # 为指定连杆逐个添加接触传感器，用于奖励与终止判断。
        # update_period=0 表示每次传感器更新时更新；history_length=6 保存力历史，
        # 但当前接触惩罚只读取 net_forces_w，而未使用历史数据。
        # debug_vis=True 会显示调试可视化，适合排查接触位置。
        contact_sensors = []
        for link_name in self.cfg.link_dof_name:
            sensor = ContactSensor(ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/mybuddy/{link_name}",
                update_period=0.0,
                history_length=6,
                debug_vis=True,
            ))
            contact_sensors.append(sensor)

        # 在 USD 场景中生成不冲突的植物材料路径，供可变形资产共享。
        self.deformable_material_path = omni.usd.get_stage_next_free_path(self.stage, "/plant_material", True)

        # 从源环境复制出 N 个并行场景，再显式设置跨环境碰撞过滤。
        # 这里只传入空的全局对象路径列表；碰撞过滤行为由场景和物理配置共同决定。
        self.scene.clone_environments(copy_from_source=True)
        self.scene.filter_collisions(global_prim_paths=[])

        # 为各环境的植物设置可变形物理、主茎固定和分支连接，返回资产列表。
        self.deformable_objects_list = add_plant(
            self.stage, self.scene.env_prim_paths, self.deformable_material_path, n_plants=1
        )

        # 注册到 InteractiveScene，使框架统一更新、重置和向仿真同步这些资产。
        # 仅在 USD 中创建对象不等于已经注册到这些资产/传感器字典。
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        for i, sensor in enumerate(contact_sensors):
            self.scene.sensors[f"contact_sensor_{i}"] = sensor
        for i, cube in enumerate(self.goal_cubes):
            self.scene.rigid_objects[f"goal_cube_{i}"] = cube
        for i, obj in enumerate(self.deformable_objects_list):
            self.scene.deformable_objects[f"stalk_{i}"] = obj

        # 一个策略步跨 decimation 个物理步：默认 (1/60)*2 = 1/30 秒。
        # 队列每项保存 N 个环境的一次可见性判断；初始全零避免立即判定成功。
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.detection_sequence = deque(
            [torch.zeros(self.num_envs, device=self.device) for _ in range(self.cfg.maxlen)],
            maxlen=self.cfg.maxlen,
        )
        self.done = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # ------------------------------------------------------------------
    # 动作执行：策略输出转换为关节位置目标
    # ------------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 每个策略步处理一次动作，输入形状为 (N,6)。
        # clone 保留原始动作，供奖励中的动作幅度惩罚使用。
        self.raw_actions = actions.clone()
        # 截断动作后乘控制步长和缩放系数，得到相对当前关节位置的增量。
        # self.actions 保存的是位置目标（弧度），而非原始动作或直接关节力矩。
        self.actions = (
            torch.clamp(self.raw_actions, -1, 1) * self.dt * self.action_scale
            + self._robot.data.joint_pos[:, self._arm_dof_idx]
        )
        # 对索引 0 对应关节的位置目标额外限幅；其他关节没有在这里显式限幅。
        self.actions[:, 0].clamp_(-1.0, 0.7)

    def _apply_action(self) -> None:
        # 在物理子步中下发位置目标，具体力矩响应由配置的关节执行器处理。
        # 一个策略步内保持同一组目标，不在每个物理子步重复累加位置增量。
        self._robot.set_joint_position_target(self.actions, joint_ids=self._arm_dof_idx)

    # ------------------------------------------------------------------
    # 观测生成：图像、关节与末端位置组成策略输入
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        # 每个策略步输出当前观测。六个受控关节的位置形状为 (N,6)，单位弧度。
        joint_readings = self._robot.data.joint_pos[:, self._arm_dof_idx]

        # RGB 先缩放到 [0,1]，再逐环境、逐颜色通道减去空间均值。
        # rgb_image_raw 保留去均值前的副本；输入网络的是去均值后的图像。
        self.rgb_image = self._tiled_camera.data.output["rgb"].clone() / 255.0
        self.rgb_image_raw = self.rgb_image.clone()
        self.rgb_image -= torch.mean(self.rgb_image, dim=(1, 2), keepdim=True)
        self.rgb_image = self.rgb_image.to(dtype=torch.float32)

        # 深度通常为 (N,H,W,1)，单位米；将正无穷替换为 0。
        # 这里没有进行深度归一化，也没有显式处理 NaN；缓存还会被奖励函数读取。
        self.depth_image = self._tiled_camera.data.output["depth"].clone()
        self.depth_image[self.depth_image == float("inf")] = 0

        # 取末端连杆 left_arm_l6 的世界质心坐标，减去根部世界质心坐标。
        # 这是世界坐标轴下的相对位移，不包含末端朝向，也没有逆旋转到基座坐标轴。
        self.ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )

        # 沿水果通道维 gather，每个环境只选自己的目标编号。
        # 将 (N,) 编号扩展成 (N,W,H,1)，输出目标掩码为单通道图像。
        cube_masks_selected = torch.gather(
            self.cube_maskas,
            3,
            self.target_cube_idx.view(-1, 1, 1, 1).expand(
                -1, self.cfg.tiled_camera.width, self.cfg.tiled_camera.height, 1
            ),
        )
        # 按最后一维拼接：RGB 3 通道 + 目标掩码 1 通道 + 深度 1 通道。
        # 默认形状 (N,256,256,5)；字段名 rgb 沿用现有接口，但并非只有 RGB。
        image_obs = torch.cat((self.rgb_image, cube_masks_selected, self.depth_image), dim=3)

        # 外层 policy 是 Isaac Lab 的观测组名，内层三个字段交给 skrl 处理。
        # 当前 Actor 与 Critic 均使用这组观测，没有 Critic 专属的额外状态输入。
        return {"policy": {
            "rgb": image_obs.to(dtype=torch.float32),
            "joints": joint_readings.to(dtype=torch.float32),
            "ee_position": self.ee_position.to(dtype=torch.float32),
        }}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        # 为每个并行环境返回一个即时奖励，最终形状为 (N,)，N = num_envs。
        # 总奖励 = 可见性连续奖励 + 达标奖励 + 持续达标奖励 + 动作惩罚 + 接触惩罚。
        # 这些是环境原始奖励；训练端是否再次缩放由算法配置决定。
        # 注意本机 Lab 2.1 的 step 顺序为：终止检查 → 奖励 → 重置 → 观测。
        # 因此这里读取的是上次 _get_observations 缓存的 depth_image，
        # 而 contact_reward 已由本步 _get_dones 更新；两者的采样时刻并不完全一致。

        # 1. 通过深度阈值生成“植物”近似掩码，形状为 (N, H, W, 1)。
        # 这是距离筛选，不是语义分割：同深度范围的其他物体也可能被计入。
        plant_mask = calculate_plant_mask(self.depth_image)

        # 2. 从所有水果的掩码中选取本回合指定目标。
        # target_cube_idx: (N,)，默认 0=红球、1=绿球；one_hot: (N, 1, 1, K)。
        # cube_maskas: (N, H, W, K)，K 是水果数量；广播相乘后沿 K 求和。
        one_hot = F.one_hot(self.target_cube_idx, num_classes=self.cfg.num_goal_cubes).float().unsqueeze(1).unsqueeze(1)
        selected_mask = torch.sum(self.cube_maskas * one_hot, dim=3, keepdim=True)
        # selected_mask: (N, H, W, 1)。与植物掩码相乘得到交集，
        # 对所有图像维度求和，得到各环境的遮挡像素数 O，形状为 (N, 1)。
        occlusion_pixels = torch.sum(selected_mask * plant_mask, dim=(1, 2, 3)).reshape(-1, 1)
        # 3. 连续可见性奖励：10 × (1 - O / 900)，O 越小奖励越大。
        # MASK_SIZE 默认 30；这里并未将奖励截断到 [0, 10]。
        # 当前掩码生成代码在区域不被边界截断时可能生成 31×31 的区域，
        # 但分母仍是 30×30，所以该比值不是精确遮挡比例，奖励也可能略小于 0。
        occlusion_reward = (1 - occlusion_pixels / (MASK_SIZE * MASK_SIZE)) * 10.0

        # 4. 阈值达标奖励：O <= 160 时额外 +3，否则为 0。
        # 名字中的 full_visibility 表示满足阈值，不表示完全无遮挡。
        full_visibility_reward = torch.where(
            occlusion_pixels <= 160,
            torch.tensor(3.0, device=self.device),
            torch.tensor(0.0, device=self.device),
        ).reshape(self.num_envs)

        # 将每个环境本次是否达标写入历史队列；该条件是布尔量 (N,)。
        # 队列长度由 cfg.maxlen 控制，默认 10；重置时对应环境的历史清零。
        full_visibility_mask = full_visibility_reward > 0
        self.detection_sequence.append(full_visibility_mask)

        # 5. 将历史堆叠为 (maxlen, N)，沿时间维检查是否连续全部达标。
        # 默认 30 Hz 下 10 次检查对应约 0.33 秒的检查窗口。
        # 持续达标奖励为 +20；代码没有“仅首次达标发放”的额外锁存条件。
        sustained_detection = torch.all(torch.stack(list(self.detection_sequence)), dim=0)
        sustained_reward = sustained_detection.float() * 20.0

        # 6. 动作惩罚：对截断后的六维原始动作取 L2 范数，再乘 -0.06。
        # 惩罚对象是策略动作，不是实际关节速度、位移或相邻动作之差。
        action_penalty = torch.norm(self.raw_actions.clamp(-1.0, 1.0), dim=1) * -0.06
        # 仅在本次可见性达标时启用，鼓励目标已可见后减少不必要运动。
        action_penalty *= full_visibility_mask.float()

        # 保存持续达标标志，供后续终止检查读取；此处不直接调用 reset。
        # 因 _get_dones 先于本函数执行，不能把此赋值理解成本步即时终止。
        self.done = sustained_detection
        occlusion_reward = occlusion_reward.reshape(self.num_envs)

        # 所有分项统一为 (N,) 后相加。接触惩罚的计算见 _calculate_contact_reward。
        # 累计回报同时受每步奖励和回合长度影响，不等同于任务成功率。
        return self.contact_reward + full_visibility_reward + sustained_reward + action_penalty + occlusion_reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 (terminated, truncated)，二者形状均为 (N,)。
        # 先更新接触标志和 -5/0 接触奖励，供随后执行的 _get_rewards 使用。
        self.contact_reward = self._calculate_contact_reward(len(self.cfg.link_dof_name))
        # 达到步数上限属于时间截断，而非达标或接触造成的任务终止。
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 任意接触或已缓存的持续达标标志都会终止，不区分接触对象类别。
        return self.in_contact | self.done, time_out

    # ------------------------------------------------------------------
    # 回合重置：仅复位指定环境，保留其他并行环境的进度
    # ------------------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # env_ids 指定本次重置的环境索引；None 表示全部。
        # 父类负责框架层面的重置，本方法继续写入任务特有的资产状态。
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        super()._reset_idx(env_ids)

        # 基于默认关节位置叠加预设姿态，然后随机设置第三个受控关节。
        # 注意 NumPy 只为本批重置抽一次分支：本批共享负区间或正区间的选择，
        # 但 sample_uniform 为各环境分别采样该区间内的角度。
        joint_pos = self._robot.data.default_joint_pos[env_ids][:, self._arm_dof_idx] + self.root_init_angles[env_ids]
        if np.random.uniform() < 0.5:
            joint_pos[:, 2] = sample_uniform(-0.52, -0.17, joint_pos[:, 2].shape, self.device)
        else:
            joint_pos[:, 2] = sample_uniform(0.2, 0.3, joint_pos[:, 2].shape, self.device)
        self.last_angles[env_ids] = joint_pos.clone()

        # 计算带环境原点偏移的机器人根状态。当前下面没有将它写回仿真，
        # 所以不要把这段局部变量计算理解为已经重置了机器人根部位姿。
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # 同时设置控制目标和实际关节状态，并清零受控关节速度。
        # 只设置目标会让机器人逐渐运动到该位置，而不能立即完成复位。
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.set_joint_position_target(joint_pos, self._arm_dof_idx, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, self._arm_dof_idx, env_ids)

        # 将本批环境的软体节点恢复为默认节点状态，清除上回合留下的形变/速度。
        for obj in self.deformable_objects_list:
            obj: DeformableObject
            nodal_state = obj.data.default_nodal_state_w.clone()[env_ids]
            obj.write_nodal_state_to_sim(nodal_state, env_ids)

        # 为每个重置环境独立抽取目标水果编号，并重置全部候选水果的位置。
        rand_idx = torch.randint(0, self.cfg.num_goal_cubes, (len(env_ids),), device=self.device)
        self.target_cube_idx[env_ids] = rand_idx

        # 从初始化时保存的世界状态出发，仅对 X 加 [0,0.15] m 的随机偏移。
        # 每回合都从基准位置采样，不是在上一回合位置上持续累加。
        base_states = self.root_cube_positions[env_ids]
        new_positions = base_states[:, :, :3].clone()
        new_positions[..., 0] += sample_uniform(0.00, 0.15, new_positions[..., 0].shape, self.device)

        # 写回球体位姿（前 7 个分量），并将其线速度和角速度置零。
        # 初始化基准来自 root_com_state_w；移植到质心偏移资产时需核对写入 API 约定。
        for k, cube in enumerate(self.goal_cubes):
            states_k = base_states[:, k, :].clone()
            states_k[:, :3] = new_positions[:, k, :3]
            cube.write_root_pose_to_sim(states_k[:, :7], env_ids)
            cube.write_root_velocity_to_sim(torch.zeros_like(states_k[:, 7:]), env_ids)

        # 重新计算本批环境中全部水果的掩码，而非只计算被选中的一个。
        # 掩码仅在这里刷新，不会在球或相机随后移动时自动更新。
        for k in range(self.cfg.num_goal_cubes):
            pos_k = new_positions[:, k, :3]
            # 将 (x,y,z) 补成齐次坐标 (x,y,z,1)，再用每个环境的投影矩阵批量投影。
            ones = torch.ones((pos_k.shape[0], 1), device=self.device)
            pts_hom = torch.cat((pos_k, ones), dim=1).unsqueeze(2)
            img_hom = torch.bmm(self.projection_matrix[env_ids], pts_hom).squeeze(2)
            # 将投影结果前两个分量除以第三个分量，得到二维像素位置。
            # 当前没有对分母接近零、目标位于相机后方等情况做额外判断。
            img_2d = img_hom[:, :2] / img_hom[:, 2:].expand(-1, 2)
            # 生成目标矩形区域，并沿图像宽度轴翻转，沿用当前投影/图像约定。
            # 这里是位置提示区域，不是水果外轮廓分割；边界由掩码工具函数裁剪。
            masks_k = torch.flip(
                create_batched_masks(
                    (self.cfg.tiled_camera.width, self.cfg.tiled_camera.height), img_2d, MASK_SIZE
                ),
                dims=[2],
            ).to(dtype=torch.float32, device=self.device)
            self.cube_maskas[env_ids, :, :, k] = masks_k

        # 刷新所选环境的 RGB 缓存；这里只读取已有相机输出，不主动触发一次新渲染。
        raw_rgb = (self._tiled_camera.data.output["rgb"].clone()[env_ids] / 255.0).to(dtype=torch.float32)
        self.rgb_image[env_ids] = raw_rgb - torch.mean(raw_rgb, dim=(1, 2), keepdim=True)

        # 保存末端相对位置的辅助缓存；当前奖励没有使用它计算位移差。
        self.last_ee_position = (
            torch.squeeze(self._robot.data.body_com_pos_w[:, self._arm_pos_idx[0]])
            - self._robot.data.root_com_pos_w
        )

        # 将历史队列暂时堆成 (maxlen,N)，只清零本批重置环境对应的列，
        # 再还原为队列，避免旧回合的可见记录带入新回合。
        # 此处没有显式清零 self.done；终止标志与奖励回调的时序需结合阅读。
        tensor_seq = torch.stack(list(self.detection_sequence))
        tensor_seq[:, env_ids] = 0
        self.detection_sequence = deque(list(tensor_seq), maxlen=self.cfg.maxlen)

    # ------------------------------------------------------------------
    # 辅助函数：汇总接触传感器并计算惩罚
    # ------------------------------------------------------------------

    def _calculate_contact_reward(self, num_sensors: int) -> torch.Tensor:
        # 读取 cfg.link_dof_name 对应的连杆接触传感器。
        # 各传感器 net_forces_w 为 (N, B_i, 3)，表示世界坐标系中的净接触力；
        # 沿刚体维拼接，得到 (N, sum(B_i), 3)。这里只使用当前读数，没有使用历史。
        readings = torch.cat(
            [self.scene.sensors[f"contact_sensor_{i}"].data.net_forces_w for i in range(num_sensors)], dim=1
        )
        # 对刚体和 XYZ 分量求 any：任意一个分量非零就标记接触。
        # 这里没有设置力阈值，也没有过滤“是否与植物接触”。
        # 因此应理解为净接触力非零检测，而非碰撞严重程度或植物损伤估计。
        self.in_contact = torch.any(readings != 0, dim=(1, 2))
        # 有接触返回 -5，无接触返回 0；接触还会在 _get_dones 中触发终止。
        return torch.where(self.in_contact, torch.tensor(-5.0, device=self.device), torch.tensor(0.0, device=self.device))
