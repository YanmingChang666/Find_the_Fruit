# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Find-the-Fruit 任务的环境配置（Config）。

本文件只"声明"参数，不包含逻辑：
  - 仿真参数（物理步长、PhysX 缓冲区）
  - 机器人（ArticulationCfg）及其关节执行器
  - 相机（TiledCameraCfg）
  - 动作 / 观测空间
  - 并行环境数量与间距
真正使用这些参数的是 find_the_fruit_env.py 中的 FindTheFruitEnv。
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .assets import ROBOT_USD

# 目标水果在图像上的"位置提示掩码"边长（像素）。
# 环境中用它生成方形掩码，奖励中也用 MASK_SIZE^2 作为遮挡比例的分母。
MASK_SIZE = 30


@configclass
class FindTheFruitEnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # 时间相关参数
    # ------------------------------------------------------------------
    # decimation：每个"策略步"包含的物理子步数。
    # 物理 dt = 1/60 s，decimation = 2 → 策略频率 30 Hz（每步 1/30 s）。
    decimation = 2
    # 单回合时长 5 秒 → 最大回合步数 ≈ 5 / (1/30) = 150 个策略步。
    episode_length_s = 5.0
    # 动作缩放系数：动作 ∈ [-1,1] 乘以 dt 和该系数，得到每步关节位置增量（rad）。
    # 因此 1.0 表示满幅动作时关节角速度上限约为 1 rad/s。
    action_scale = 1.0
    # 不使用非对称 Critic 的特权状态（Actor/Critic 看同一份观测）。
    state_space = 0

    # ------------------------------------------------------------------
    # 仿真（物理引擎）配置
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        # 每 decimation 个物理步渲染一次，与策略步对齐，节省渲染开销。
        render_interval=decimation,
        physx=PhysxCfg(
            # GPU 临时缓冲区 / 碰撞对容量：软体（植物）+ 多环境并行时需要调大，
            # 否则 PhysX 会报缓冲区溢出。
            gpu_temp_buffer_capacity=2**26,
            gpu_found_lost_pairs_capacity=2**27,
            # 位置迭代次数下限设得很高（128），提高可变形体与连接约束的稳定性，
            # 代价是仿真速度下降。
            min_position_iteration_count=128,
        ),
        gravity=(0.0, 0.0, -9.81),
    )

    # ------------------------------------------------------------------
    # 机器人配置
    # ------------------------------------------------------------------
    robot_cfg = ArticulationCfg(
        # env_.* 是正则占位，每个并行环境下都会生成一个 mybuddy。
        prim_path="/World/envs/env_.*/mybuddy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            # 需要开启，否则后面的 ContactSensor 读不到接触力。
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # 关闭重力：机械臂靠位置控制，避免重力下垂干扰。
                disable_gravity=True,
                # 限制穿透恢复速度，防止碰撞后"弹飞"。
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                # 允许自碰撞（机械臂连杆之间）。
                enabled_self_collisions=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        # 执行器：隐式 PD 位置控制（由 PhysX 内部求解，稳定且快）。
        # ".*" 匹配所有关节；目标 = stiffness*(q_target - q) - damping*q_dot。
        actuators={
            ".*": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=1000.0,
                velocity_limit=10.0,
                stiffness=800.0,
                damping=4.0,
            ),
        },
    )

    # 受策略控制的 6 个关节：1 个基座旋转关节 + 左臂 5 个关节。
    # 动作维度 = len(arm_dof_name) = 6。
    arm_dof_name = [
        "bc2bl",
        "left_arm_j1",
        "left_arm_j2",
        "left_arm_j3",
        "left_arm_j4",
        "left_arm_j5",
    ]
    # 需要挂接触传感器的连杆（环境中会为每个连杆创建一个 ContactSensor）。
    # 其中 left_arm_l6 同时也是末端执行器（相机观察点/末端位置的来源）。
    link_dof_name = ["left_arm_l1", "left_arm_l3", "left_arm_l4", "left_arm_l5", "left_arm_l6"]

    # ------------------------------------------------------------------
    # 相机配置（TiledCamera：多环境批量渲染，吞吐远高于逐个相机）
    # ------------------------------------------------------------------
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        # 注意：相机挂在每个环境根下而不是机器人上，是固定视角相机。
        prim_path="/World/envs/env_.*/Camera",
        # 相对环境原点的位姿；rot 为四元数 (w, x, y, z)，约定为 opengl 相机坐标系。
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.5), rot=(0.0, 0.0, 0.53833, 0.84274), convention="opengl"),
        # 同时输出 RGB 和深度，两者在观测里被拼成 5 通道图像。
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            # 针孔相机内参：焦距 4mm、水平光圈 5.76mm → 水平视场角约 71°。
            focal_length=4.0,
            focus_distance=0.6,
            horizontal_aperture=5.760000228881836,
            vertical_aperture=3.5999999046325684,
            # 近/远裁剪面（米）。近裁剪 0.076 m；超出 10 m 的深度为 inf。
            clipping_range=(0.076, 10.0),
            f_stop=240.0,
        ),
        width=256,
        height=256,
    )

    # 候选目标水果（球体）的数量 K。每回合随机指定其中一个为目标。
    num_goal_cubes: int = 2

    # ------------------------------------------------------------------
    # 动作 / 观测空间
    # ------------------------------------------------------------------
    # 动作：6 维连续量，范围 [-1, 1]，经环境换算为关节位置增量。
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(len(arm_dof_name),))
    # 观测：Dict 空间，与 _get_observations 返回的 "policy" 字典一一对应。
    #   joints      : 6 个受控关节角度（rad）
    #   rgb         : (H, W, 5) = RGB(3) + 目标水果掩码(1) + 深度(1)
    #                 （字段名叫 rgb 只是沿用接口，实际是 5 通道）
    #   ee_position : 末端相对机器人根部的世界坐标位移 (3,)
    # 注意：这里的 low/high 仅用于声明，并不会真正对数据做截断。
    observation_space = {
        "joints": gym.spaces.Box(low=-20.00, high=20.0, shape=(len(arm_dof_name),), dtype=np.float32),
        "rgb": gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(tiled_camera.height, tiled_camera.width, 5),
            dtype=np.float32,
        ),
        "ee_position": gym.spaces.Box(low=-20.00, high=20.0, shape=(3,), dtype=np.float32),
    }

    # ------------------------------------------------------------------
    # 场景（并行环境）
    # ------------------------------------------------------------------
    # num_envs=50：并行环境数（可被命令行 --num_envs 覆盖）。
    # env_spacing=2.5：各环境原点间隔（米）。
    # replicate_physics=False：因为有可变形植物 + 各环境独立的 PhysX 连接，
    #   不能让 PhysX 共享同一份物理模板，必须逐环境独立创建（更慢但必需）。
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=50, env_spacing=2.5, replicate_physics=False)

    # ------------------------------------------------------------------
    # 任务自定义参数
    # ------------------------------------------------------------------
    # max_y_pos 与 rew_scale_alive 在当前环境代码中并未被使用（遗留参数）。
    max_y_pos = 0.0
    # 连续可见性判定窗口：detection_sequence 队列长度（10 步 ≈ 0.33 s）。
    maxlen = 10
    rew_scale_alive = -0.1