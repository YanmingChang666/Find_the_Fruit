# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for building the Find-the-Fruit scene: deformable plant setup and physics attachments.

本文件负责"植物"这一最复杂的场景元素，分三步（由 add_plant 统一调度）：
  1. spawn_plant            ：把植物 USD 几何加载进场景（在 env 中被先调用）
  2. make_deformable        ：给植物的每一块网格加上可变形体（软体）物理和材料
  3. attach_cylinder_to_ground：用 PhysX 附着（Attachment）约束把主茎固定到地面，
                               并把 5 个分枝固定到主茎上
植物 USD 的结构约定（代码依赖，改资产时需同步检查）：
  Plant{n}/
    ├─ <第 0 个子节点>      ：被跳过（[1:] 切片），通常是材质/Looks 等非几何节点
    ├─ stalk   /plant_023   ：主茎网格（分枝要附着到它上面，路径写死）
    └─ stalk1..stalk5       ：5 个分枝，每个子节点下第一个孩子是实际网格
"""

from __future__ import annotations

import omni.usd
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg
from isaaclab.utils.math import quat_from_euler_xyz

import torch


def spawn_plant(stage_func_target: str, translation: tuple, usd_path: str, device: str) -> None:
    """Spawn the plant USD at the given prim path with a fixed pose."""
    import torch as _torch
    # 旋转全 0 的欧拉角 → 单位四元数 (w,x,y,z)=(1,0,0,0)，即不旋转。
    # 保留欧拉角写法是为了以后方便调整植物朝向。
    euler = _torch.tensor([0.0, 0.0, 0.0], device=device)
    quat = quat_from_euler_xyz(*euler).cpu().numpy()
    # scale=0.05：原始植物模型较大，缩小到 5% 以匹配机械臂工作空间（米）。
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path, scale=(0.05, 0.05, 0.05))
    # stage_func_target 带 env_.* 通配符，会在所有并行环境中各生成一份。
    cfg.func(
        stage_func_target,
        cfg,
        translation=translation,
        orientation=(quat[0], quat[1], quat[2], quat[3]),
    )


def make_deformable(stage, deformable_material_path: str, prim_dict: dict, simulation_resolution: int = 10) -> None:
    """Apply deformable-body physics to plant mesh prims and assign material."""
    # prim_dict：{子节点名: 网格 prim}，第一项是主茎，其余是分枝。
    key, value = list(prim_dict.items())[0]
    # 先创建一份"可变形体材料"（所有植物部件共用同一路径）：
    #   杨氏模量 7.5e8 Pa → 比较硬，像木质茎秆；泊松比 0.49 → 近似不可压缩；
    #   damping_scale 控制内部阻尼，density 为密度。
    deformableUtils.add_deformable_body_material(
        stage,
        deformable_material_path,
        youngs_modulus=7.5e8,
        poissons_ratio=0.49,
        damping_scale=1.0,
        dynamic_friction=0.5,
        density=2.0,
    )
    # 把主茎网格转成 PhysX 可变形体：
    #   collision_simplification：自动简化碰撞网格，降低计算量；
    #   simulation_hexahedral_resolution：生成六面体仿真网格的分辨率，越大越精细越慢；
    #   solver_position_iteration_count=128：高迭代次数提高软体稳定性；
    #   self_collision=False：不计算植物自碰撞，节省开销。
    deformableUtils.add_physx_deformable_body(
        stage,
        value.GetPath(),
        collision_simplification=True,
        simulation_hexahedral_resolution=simulation_resolution,
        solver_position_iteration_count=128,
        self_collision=False,
    )
    # 把上面创建的材料绑定到主茎。
    physicsUtils.add_physics_material_to_prim(stage, value.GetPrim(), deformable_material_path)

    # 对 5 个分枝重复同样的软体设置，并共用同一材料。
    for _, value in list(prim_dict.items())[1:]:
        deformableUtils.add_physx_deformable_body(
            stage,
            value.GetPath(),
            collision_simplification=True,
            simulation_hexahedral_resolution=simulation_resolution,
            solver_position_iteration_count=128,
            self_collision=False,
        )
        physicsUtils.add_physics_material_to_prim(stage, value.GetPrim(), deformable_material_path)


def attach_cylinder_to_ground(stage, prim_dict: dict, prim_name: str) -> None:
    """Attach the base stalk to the ground plane and sub-stalks to the main stalk via PhysX attachments."""
    # ---- 1) 主茎 ↔ 地面：固定植物根部，否则软体会掉落/漂走 ----
    key, value = list(prim_dict.items())[0]
    attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    # actor0 = 软体（主茎），actor1 = 地面碰撞面（刚体）。
    att.GetActor0Rel().SetTargets([value.GetPath()])
    att.GetActor1Rel().SetTargets(["/World/ground/GroundPlane/CollisionPlane"])
    # 自动附着（AutoAttachment）：由 PhysX 根据几何重叠自动生成附着点，
    # 无需手工指定顶点。下面是各自动附着参数：
    api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    # 软体顶点与对方表面距离在此偏移内才会被附着（米）。
    api.GetPrim().GetAttribute("physxAutoAttachment:deformableVertexOverlapOffset").Set(0.05)
    # 开启"软体顶点附着"与"刚体表面附着"。
    api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableVertexAttachments").Set(True)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableRigidSurfaceAttachments").Set(True)
    # 碰撞过滤：被附着在一起的两物体之间不再相互碰撞，避免附着区域互相"顶开"。
    api.GetPrim().GetAttribute("physxAutoAttachment:enableCollisionFiltering").Set(True)
    api.GetPrim().GetAttribute("physxAutoAttachment:collisionFilteringOffset").Set(0.01)
    api.GetPrim().GetAttribute("physxAutoAttachment:enableDeformableFilteringPairs").Set(True)

    # ---- 2) 分枝 ↔ 主茎：把每个分枝固定到主茎的网格上 ----
    # 注意主茎网格路径 "{prim_name}/stalk/plant_023" 是写死的，换植物资产时要同步修改。
    for key, value in list(prim_dict.items())[1:]:
        attachment_path = value.GetPath().AppendElementString(f"attachment_{key}")
        att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
        att.GetActor0Rel().SetTargets([value.GetPath()])
        att.GetActor1Rel().SetTargets([f"{prim_name}/stalk/plant_023"])
        api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
        # 分枝与主茎连接的容差更小（0.005 m），只有紧贴的顶点才会被连起来。
        api.GetPrim().GetAttribute("physxAutoAttachment:deformableVertexOverlapOffset").Set(0.005)


def add_plant(stage, env_prim_paths: list[str], deformable_material_path: str, n_plants: int = 1) -> list:
    """Register deformable plant objects and configure their physics for all envs.

    Returns a flat list of DeformableObject instances (stalk + 5 sub-stalks per plant).
    """
    deformable_objects: list[DeformableObject] = []

    for plant_number in range(1, n_plants + 1):
        # ---- A) 在 Isaac Lab 层注册软体对象（用于读写节点状态） ----
        # prim_path 带 env_.* 通配，一个 DeformableObject 管理所有并行环境中的同名软体。
        # 每株植物 = 1 个主茎 + 5 个分枝，共 6 个 DeformableObject。
        # 环境 _reset_idx 里会遍历它们，把节点状态恢复为默认值。
        stalk_cfg = DeformableObjectCfg(
            prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk",
            debug_vis=True,
        )
        deformable_objects.append(DeformableObject(stalk_cfg))

        for i in range(1, 6):
            sub_cfg = DeformableObjectCfg(
                prim_path=f"/World/envs/env_.*/Plant{plant_number}/stalk{i}",
                debug_vis=True,
            )
            deformable_objects.append(DeformableObject(sub_cfg))

        # ---- B) 在 USD 层逐环境设置物理（软体 + 附着） ----
        # 这些是 USD/PhysX 属性设置，必须对每个环境的 prim 单独执行。
        for env_path in env_prim_paths:
            plant_prim = stage.GetPrimAtPath(f"{env_path}/Plant{plant_number}")
            # 每个子节点（stalk / stalkN）下第一个孩子是真正的网格。
            plant_meshes = [mesh.GetAllChildren()[0] for mesh in plant_prim.GetAllChildren()]
            # 跳过第 0 个子节点（非植物几何节点）。
            plant_meshes = plant_meshes[1:]
            prim_dict = dict(zip(plant_prim.GetAllChildrenNames()[1:], plant_meshes))
            make_deformable(stage, deformable_material_path, prim_dict)
            attach_cylinder_to_ground(stage, prim_dict, f"{env_path}/Plant{plant_number}")

    return deformable_objects