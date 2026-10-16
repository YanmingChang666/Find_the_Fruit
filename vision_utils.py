# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: F401  (keep for downstream callers)


def quaternion_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to rotation matrices (N, 3, 3)."""
    if quat.dim() == 1:
        quat = quat.unsqueeze(0)
    quat = quat / quat.norm(dim=1, keepdim=True)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=1,
    ).reshape(-1, 3, 3)


def create_batched_masks(
    image_size: tuple[int, int],
    cube_pixel_locations: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Return binary masks (n_envs, H, W) with an n×n block set to 1 at each cube pixel location."""
    n_envs = cube_pixel_locations.shape[0]
    height, width = image_size
    masks = torch.zeros((n_envs, height, width), dtype=torch.float32, device=cube_pixel_locations.device)
    px = cube_pixel_locations[:, 0]
    py = cube_pixel_locations[:, 1]
    # 方形目标区域以投影像素为中心；端点裁剪到图像内。
    # 切片右端不包含在区域内，但下面又额外 +1，因此偶数 n=30
    # 在内部区域通常生成 31×31 的掩码，而不是文档字符串所称的 n×n。
    # 奖励函数仍以 MASK_SIZE² 归一化；这里仅说明现状，不修改算法。
    half = n // 2
    x0 = torch.clamp(px - half, 0, width - 1).int()
    x1 = torch.clamp(px + half + 1, 0, width).int()
    y0 = torch.clamp(py - half, 0, height - 1).int()
    y1 = torch.clamp(py + half + 1, 0, height).int()
    for i in range(n_envs):
        masks[i, y0[i]:y1[i], x0[i]:x1[i]] = 1.0
    return masks


@torch.jit.script
def calculate_plant_mask(depth_obs: torch.Tensor) -> torch.Tensor:
    """Return a float mask selecting depth values corresponding to the plant (0.1–0.4 m)."""
    # 输入通常为 (N, H, W, 1)，深度单位为米；输出保持同形状。
    # 区间两端均包含：0.1 <= depth <= 0.4 的像素置 1，其余置 0。
    # 这是利用场景距离范围构造的近似掩码，并不判断像素语义。
    # 机械臂、水果或其他物体若落入该深度区间，也可能被计入。
    # 奖励函数将其与目标水果掩码相乘，再求和得到近似遮挡像素数。
    # clone 创建副本，detach 切断梯度，float 转为浮点以便乘法和求和。
    return ((depth_obs >= 0.1) & (depth_obs <= 0.4)).clone().detach().float()
