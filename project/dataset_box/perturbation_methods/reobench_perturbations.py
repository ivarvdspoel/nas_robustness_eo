import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import functional as TF


# def _ensure_cpu_float_tensor(x):
#     if not isinstance(x, torch.Tensor):
#         raise TypeError(f"Expected torch.Tensor, got {type(x)}")
#     return x.detach().to(device="cpu", dtype=torch.float32)


def add_gaussian_noise_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    sigma_levels = [0.04, 0.05, 0.06, 0.07, 0.08]
    sigma = sigma_levels[severity - 1]
    noise = torch.randn_like(x)
    x_noisy = torch.clamp(x + noise * sigma, 0.0, 1.0)
    return x_noisy


def add_salt_pepper_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    amount_levels = [0.005, 0.01, 0.02, 0.03, 0.05]
    amount = amount_levels[severity - 1]

    mask = torch.rand_like(x) < amount
    salt_or_pepper = torch.randint(0, 2, x.shape, device="cpu", dtype=torch.int64).to(x.dtype)

    x_noisy = x.clone()
    x_noisy[mask] = salt_or_pepper[mask]
    return x_noisy


def gaussian_blur_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    kernel_sizes = [3, 5, 7, 9, 11]
    k = kernel_sizes[severity - 1]
    pad = k // 2

    _, C, _, _ = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    kernel = torch.ones((C, 1, k, k), device="cpu", dtype=x.dtype) / (k * k)
    blurred = F.conv2d(x_pad, kernel, groups=C)
    return blurred


def motion_blur_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    kernel_sizes = [3, 5, 7, 9, 11]
    k = kernel_sizes[severity - 1]

    _, C, _, _ = x.shape
    kernel = torch.zeros((C, 1, k, k), device=x.device, dtype=x.dtype)
    kernel[:, :, k // 2, :] = 1.0 / k

    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(x_pad, kernel, groups=C)
    return blurred


def adjust_brightness_contrast_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    brightness_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    contrast_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
    b = brightness_levels[severity - 1]
    c = contrast_levels[severity - 1]
    x_adj = torch.clamp(x * c + b, 0.0, 1.0)
    return x_adj


def add_clouds_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    cloud_density_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
    cloud_density = cloud_density_levels[severity - 1]

    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H, W, device="cpu", dtype=x.dtype)
    mask = (mask > (1.0 - cloud_density)).to(x.dtype)

    mask_blurred = []
    for i in range(B):
        mask_i = TF.gaussian_blur(mask[i:i+1], kernel_size=11)
        mask_blurred.append(mask_i)
    mask = torch.cat(mask_blurred, dim=0)

    cloud_color = torch.full_like(x, 0.9, device="cpu")
    return torch.clamp(x * (1 - mask) + cloud_color * mask, 0.0, 1.0)


def add_haze_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    intensity_levels = [0.2, 0.3, 0.4, 0.5, 0.6]
    intensity = intensity_levels[severity - 1]
    haze_layer = torch.ones_like(x, device="cpu")
    return torch.clamp(x * (1 - intensity) + haze_layer * intensity, 0.0, 1.0)


def create_data_gaps_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    lst_stripes = [2, 3, 4, 5, 6]
    lst_width = [3, 4, 5, 6, 7]
    num_stripes = lst_stripes[severity - 1]
    stripe_width = lst_width[severity - 1]

    B, C, H, W = x.shape
    angles = torch.rand(B, device="cpu", dtype=x.dtype) * np.pi
    processed = []

    yy, xx = torch.meshgrid(
        torch.arange(H, device="cpu", dtype=x.dtype),
        torch.arange(W, device="cpu", dtype=x.dtype),
        indexing="ij"
    )

    spacing = np.sqrt(H**2 + W**2) / num_stripes

    for i in range(B):
        coords = xx * torch.cos(angles[i]) + yy * torch.sin(angles[i])
        stripe_mask = (coords % spacing < stripe_width).to(x.dtype)
        processed.append(x[i] * (1 - stripe_mask.unsqueeze(0)))

    return torch.stack(processed, dim=0)


def add_compression_artifacts_batch_tensor(x, severity=1):
    #x = _ensure_cpu_float_tensor(x)
    quality_levels = [30, 25, 20, 15, 10]
    q = quality_levels[severity - 1]
    x_scaled = torch.clamp(x * 255, 0, 255)
    x_quant = torch.round(x_scaled / q) * q / 255
    return torch.clamp(x_quant, 0.0, 1.0)