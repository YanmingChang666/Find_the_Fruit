"""深度图相关工具（目前环境主文件 find_the_fruit_env.py 并未导入本文件）。

包含两部分：
  1. DepthEstimator        ：用 MiDaS 单目深度估计模型从 RGB 预测相对深度（0~1）
  2. add_depth_noise       ：对仿真深度图叠加真实深度相机常见的噪声，
                             用于 sim-to-real 的域随机化（domain randomization）
     draw_random_vertical_lines：add_depth_noise 的子函数，画随机竖直亮线
"""

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.hub import load
import torch.nn.functional as F


class DepthEstimator:
    """基于 MiDaS 的单目深度估计器：输入 RGB 批量图像，输出归一化深度。"""

    def __init__(self, model_type='MiDaS_small', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # 通过 torch.hub 下载/加载预训练模型（首次运行需要联网）。
        self.model = load("intel-isl/MiDaS", model_type).to(self.device)
        self.model.eval()
        # 预处理：缩放到 384x384，并用 ImageNet 均值/方差归一化。
        # 注意：这里直接作用于 tensor，输入应为 (N,3,H,W) 且数值范围 [0,1]。
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_batch(self, images):
        # images = torch.stack(images)
        return self.transform(images).to(self.device)

    def estimate_depth(self, images):
        """输入 (N,3,H,W)，输出 (N,H,W,1)，每张图独立归一化到 [0,1]。

        注意：MiDaS 输出的是"相对逆深度"（越大越近），不是以米为单位的真实深度；
        且每张图单独做 min-max 归一化，所以不同图之间数值不可直接比较。
        """
        with torch.no_grad():
            original_size = images.shape[-2:]  # 假设批内所有图像大小一致
            input_batch = self.preprocess_batch(images)
            depth_maps = self.model(input_batch)

            # 把模型输出（384x384）双三次插值回原始分辨率
            depth_maps_resized = F.interpolate(
                depth_maps.unsqueeze(1), size=original_size, mode="bicubic", align_corners=False
            ).squeeze(1)

            # 逐图 min-max 归一化到 [0,1]
            depth_min = depth_maps_resized.amin(dim=(1, 2), keepdim=True)
            depth_max = depth_maps_resized.amax(dim=(1, 2), keepdim=True)
            depth_normalized = (depth_maps_resized - depth_min) / (depth_max - depth_min)
        # (N,H,W) -> (N,1,H,W) -> (N,H,W,1)，与环境中深度图的布局一致
        return depth_normalized.unsqueeze(1).permute(0, 2, 3, 1)


# import torch
# import torch.nn.functional as F
# from typing import Optional

@torch.jit.script
def draw_random_vertical_lines(
    depth_tensor: torch.Tensor,
    num_lines: int,
    min_thickness: int,
    max_thickness: int,
    line_value: float
) -> torch.Tensor:
    """
    Draws `num_lines` random vertical lines on a single-channel depth image (H, W),
    with random line thickness between [min_thickness, max_thickness].

    Modifies the tensor in-place.

    Args:
        depth_tensor (torch.Tensor): Single depth image (H, W).
        num_lines (int): How many random vertical lines to draw.
        min_thickness (int): Minimum line thickness in pixels.
        max_thickness (int): Maximum line thickness in pixels.
        line_value (float): Pixel value for the lines (e.g., "white" depth).

    Returns:
        torch.Tensor: Modified depth image with vertical lines.
    """
    H = depth_tensor.shape[0]
    W = depth_tensor.shape[1]

    # 模拟某些深度相机（如结构光/ToF）在边缘或反光处产生的竖直"条纹伪影"。
    for _i in range(num_lines):
        # 随机 x 位置（线的中心列）
        x_rand = torch.randint(low=0, high=W, size=(1,))
        x = int(x_rand[0].item())

        # 随机线宽（像素）
        thickness_rand = torch.randint(low=min_thickness, high=max_thickness + 1, size=(1,))
        line_thickness = int(thickness_rand[0].item())

        # 随机竖直起止行
        y1_rand = torch.randint(low=0, high=H, size=(1,))
        y1 = int(y1_rand[0].item())
        y2_rand = torch.randint(low=0, high=H, size=(1,))
        y2 = int(y2_rand[0].item())

        # 保证 y1 <= y2
        if y2 < y1:
            temp = y1
            y1 = y2
            y2 = temp

        # 根据线宽计算水平范围，并裁剪到图像内
        x_min = max(0, x - line_thickness // 2)
        x_max = min(W, x + line_thickness // 2 + 1)

        # 竖直方向裁剪
        y1_clamped = max(0, y1)
        y2_clamped = min(H - 1, y2)

        if y2_clamped >= y1_clamped:
            # 原地写入：该区域深度被替换成 line_value
            depth_tensor[y1_clamped : y2_clamped + 1, x_min : x_max] = line_value

    return depth_tensor


@torch.jit.script
def add_depth_noise(
    depth_image: torch.Tensor,
    gaussian_std: float = 0.05,
    alpha: float = 0.001,
    beta: float = 0.002,
    gamma: float = 0.001,
    drop_prob: float = 0.05,
    speckle_var: float = 0.01,
    edge_drop_prob: float = 0.9,
    blur_kernel: int = 5,
    edge_dilation: int = 3,
    blur_iterations: int = 2,
    add_white_lines: bool = True,
    max_lines_per_image: int = 5,
    min_line_thickness: int = 1,
    max_line_thickness: int = 7
) -> torch.Tensor:
    """
    TorchScript-compatible pipeline that adds:
      - Gaussian noise
      - Depth-dependent noise
      - Edge-based dropout
      - Speckle noise
      - Optional random vertical white lines (with varying thickness)
      - Gaussian blur

    Args:
        depth_image (torch.Tensor): (N, 1, H, W) depth images.
        gaussian_std (float): Std dev of Gaussian noise.
        alpha, beta, gamma (float): Coeffs for depth-dependent noise.
        drop_prob (float): Probability of dropout (NaN).
        speckle_var (float): Variance for speckle (multiplicative) noise.
        edge_drop_prob (float): Probability for edge dropout.
        blur_kernel (int): Kernel size for Gaussian blur (must be odd).
        edge_dilation (int): Kernel size for dilating edges.
        blur_iterations (int): Number of consecutive blur passes.
        add_white_lines (bool): Whether to add vertical white lines.
        max_lines_per_image (int): Max number of white lines per image.
        min_line_thickness (int): Minimum thickness of each line.
        max_line_thickness (int): Maximum thickness of each line.

    Returns:
        torch.Tensor: Noisy depth images (N, 1, H, W).
    """
    shape = depth_image.shape
    N = shape[0]
    C = shape[1]
    H = shape[2]
    W = shape[3]
    if C != 1:
        raise RuntimeError("depth_image must have shape (N, 1, H, W).")

    # 注意：本函数要求通道在第 1 维 (N,1,H,W)，而环境里的深度图是 (N,H,W,1)，
    # 若要接入环境，需要先 permute(0,3,1,2)，处理完再 permute 回去。

    # 1. 高斯噪声：与深度无关的传感器底噪
    noise_gaussian = torch.randn_like(depth_image) * gaussian_std
    depth_noisy = depth_image + noise_gaussian

    # 2. 深度相关噪声：标准差 = alpha*z^2 + beta*z + gamma
    #    （远处点噪声更大，符合双目/ToF 相机的典型误差模型）
    std_depth = alpha * depth_image * depth_image + beta * depth_image + gamma
    noise_depth = torch.randn_like(depth_image) * std_depth
    depth_noisy = depth_noisy + noise_depth

    # 3. 边缘检测：对原始深度做 Sobel 算子，找出深度突变处（物体轮廓）
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=depth_image.device).reshape(1,1,3,3)
    sobel_y = sobel_x.transpose(2,3)

    grad_x = F.conv2d(depth_image, sobel_x, padding=1)  # (N,1,H,W)
    grad_y = F.conv2d(depth_image, sobel_y, padding=1)  # (N,1,H,W)
    grad_mag = torch.sqrt(grad_x*grad_x + grad_y*grad_y)

    # 每张图独立把梯度幅值归一化到 [0,1]
    grad_flat = grad_mag.view(N, -1)
    max_vals, _ = torch.max(grad_flat, dim=1, keepdim=True)
    max_vals = max_vals.unsqueeze(-1)  # shape => (N,1,1)
    eps = 1e-6
    edges_norm = grad_mag / (max_vals.view(N, 1, 1, 1) + eps)  # [0..1]

    # 4. 边缘膨胀：用最大池化把边缘加粗，模拟轮廓附近更宽的无效区
    if edge_dilation > 1:
        edges_norm = F.max_pool2d(edges_norm, kernel_size=edge_dilation, stride=1,
                                  padding=edge_dilation // 2)

    # 5. 数据缺失（深度空洞）：边缘处按 edge_drop_prob 丢失 + 全图随机丢失 drop_prob
    #    被丢弃的像素先置为 NaN，下一步再补洞。
    prob_map = edges_norm * edge_drop_prob
    rnd_for_edges = torch.rand_like(prob_map)
    keep_edges = (rnd_for_edges > prob_map)

    rnd_for_dropout = torch.rand_like(depth_noisy)
    keep_rand = (rnd_for_dropout > drop_prob)

    keep_mask = keep_edges & keep_rand
    depth_noisy = torch.where(keep_mask, depth_noisy, torch.tensor(float('nan'), device=depth_noisy.device))

    # 6. 斑点噪声（乘性噪声）：扰动幅度与深度值成正比
    noise_speckle = torch.randn_like(depth_noisy) * speckle_var * depth_noisy
    depth_noisy = depth_noisy + noise_speckle

    # 7. 用局部均值填补 NaN：先把 NaN 置 0，再做 3x3 平均池化，
    #    只把结果写回 NaN 位置。（注意：邻域内的空洞会把均值拉低）
    nan_mask = torch.isnan(depth_noisy)
    if torch.any(nan_mask):
        zero_filled = torch.where(nan_mask, torch.zeros_like(depth_noisy), depth_noisy)
        depth_filled = F.avg_pool2d(zero_filled, kernel_size=3, stride=1, padding=1)
        depth_noisy = torch.where(nan_mask, depth_filled, depth_noisy)

    # 8. 高斯模糊：构造高斯核，连续卷积 blur_iterations 次，模拟相机低通/分辨率限制
    if blur_kernel > 1 and (blur_kernel % 2) == 1:
        sigma = float(blur_kernel) / 6.0
        coords = torch.arange(blur_kernel, device=depth_image.device) - (blur_kernel // 2)
        coords_x = coords.unsqueeze(0)
        coords_y = coords.unsqueeze(1)
        kernel_2d = torch.exp(-(coords_x*coords_x + coords_y*coords_y) / (2.0*sigma*sigma))
        kernel_2d = kernel_2d / torch.sum(kernel_2d)
        gauss_kernel = kernel_2d.view(1,1,blur_kernel,blur_kernel)

        for _j in range(blur_iterations):
            depth_noisy = F.conv2d(depth_noisy, gauss_kernel, padding=blur_kernel // 2)

    # 9. （可选）随机竖直亮线：模拟条纹伪影。
    #    线值取该图最大深度的 1.1 倍（"比最远处还远"的亮线）。
    if add_white_lines and max_lines_per_image > 0:
        for i in range(N):
            # 取出第 i 张图的 (H,W) 视图；绘线函数会原地修改它，进而修改 depth_noisy
            image_2d = depth_noisy[i, 0, :, :]
            local_max = float(torch.max(image_2d))
            line_value = local_max * 1.1 if local_max > 0.0 else 1.0

            # 每张图随机画 [0, max_lines_per_image] 条线
            num_lines_rand = torch.randint(low=0, high=max_lines_per_image+1, size=(1,))
            num_lines = int(num_lines_rand[0].item())

            draw_random_vertical_lines(
                image_2d,
                num_lines=num_lines,
                min_thickness=min_line_thickness,
                max_thickness=max_line_thickness,
                line_value=line_value
            )

    return depth_noisy