import torch
import torch.nn.functional as F
import numpy as np

def add_gaussian_noise_batch_tensor(x, severity=1):
    """
    x: (B, C, H, W), float tensor in [0,1]
    """
    sigma_levels = [0.04, 0.05, 0.06, 0.07, 0.08]
    sigma = sigma_levels[severity - 1]
    noise = torch.randn_like(x) * sigma
    x_noisy = torch.clamp(x + noise, 0.0, 1.0)
    return x_noisy


def add_salt_pepper_batch_tensor(x, severity=1):
    amount_levels = [0.005, 0.01, 0.02, 0.03, 0.05]
    amount = amount_levels[severity - 1]

    B, C, H, W = x.shape
    mask = torch.rand_like(x) < amount
    salt_or_pepper = torch.randint_like(x, 0, 2, dtype=torch.float32)
    x_noisy = x.clone()
    x_noisy[mask] = salt_or_pepper[mask].float()
    return x_noisy



def gaussian_blur_batch_tensor(x, severity=1):
    kernel_sizes = [3,5,7,9,11]
    k = kernel_sizes[severity-1]
    pad = k // 2

    # Apply depthwise conv for each channel
    B, C, H, W = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    kernel = torch.ones((C, 1, k, k), device=x.device) / (k*k)
    blurred = F.conv2d(x_pad, kernel, groups=C)
    return blurred



def motion_blur_batch_tensor(x, severity=1):
    kernel_sizes = [2,4,6,8,10]
    k = kernel_sizes[severity-1]

    B, C, H, W = x.shape
    device = x.device
    x = x.cpu()
    kernel = torch.zeros((C, 1, k, k))
    kernel[:, :, k//2, :] = 1.0 / k
    pad = k // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    blurred = F.conv2d(x_pad, kernel, groups=C)
    return blurred.to(device)



def adjust_brightness_contrast_batch_tensor(x, severity=1):
    brightness_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    contrast_levels   = [1.0, 0.8, 0.6, 0.4, 0.2]
    b = brightness_levels[severity-1]
    c = contrast_levels[severity-1]
    x_adj = torch.clamp(x * c + b, 0.0, 1.0)
    return x_adj



from torchvision.transforms import functional as TF

def add_clouds_batch_tensor(x, severity=1):
    cloud_density_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
    cloud_density = cloud_density_levels[severity-1]

    B, C, H, W = x.shape
    mask = torch.rand(B, 1, H, W, device=x.device)
    mask = (mask > (1.0 - cloud_density)).float()
    
    # Apply gaussian blur per image in batch
    mask_blurred = []
    for i in range(B):
        mask_i = TF.gaussian_blur(mask[i:i+1], kernel_size=11)
        mask_blurred.append(mask_i)
    mask = torch.cat(mask_blurred, dim=0)
    
    cloud_color = torch.full_like(x, 0.9)
    return torch.clamp(x * (1 - mask) + cloud_color * mask, 0.0, 1.0)

def add_haze_batch_tensor(x, severity=1):
    intensity_levels = [0.2,0.3,0.4,0.5,0.6]
    intensity = intensity_levels[severity-1]
    haze_layer = torch.ones_like(x)
    return torch.clamp(x * (1 - intensity) + haze_layer * intensity, 0.0, 1.0)


def create_data_gaps_batch_tensor(x, severity=1):
    lst_stripes = [2,3,4,5,6]
    lst_width = [3,4,5,6,7]
    num_stripes = lst_stripes[severity-1]
    stripe_width = lst_width[severity-1]

    B, C, H, W = x.shape
    angles = torch.rand(B, device=x.device) * np.pi
    processed = []

    yy, xx = torch.meshgrid(torch.arange(H, device=x.device),
                            torch.arange(W, device=x.device),
                            indexing='ij')

    for i in range(B):
        coords = xx * torch.cos(angles[i]) + yy * torch.sin(angles[i])
        stripe_mask = (coords % (np.sqrt(H**2 + W**2)/num_stripes) < stripe_width).float()
        processed.append(x[i] * (1 - stripe_mask))
    return torch.stack(processed)



def add_compression_artifacts_batch_tensor(x, severity=1):
    # Simulate JPEG by adding quantization noise
    quality_levels = [30,25,20,15,10]
    q = quality_levels[severity-1]
    x_scaled = torch.clamp(x * 255, 0, 255)
    x_quant = torch.round(x_scaled / q) * q / 255
    return torch.clamp(x_quant, 0.0, 1.0)