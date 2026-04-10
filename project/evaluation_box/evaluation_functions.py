# File with robustness evaluation metrics found in the literature
import torch



def compute_miou_(preds, labels, num_classes=4):
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for cls in range(num_classes):
        inter = ((preds == cls) & (labels == cls)).sum()
        uni = ((preds == cls) | (labels == cls)).sum()
        intersection[cls] += inter
        union[cls] += uni

    return (intersection / union).mean().item()

def compute_miou_per_sample(preds, labels, num_classes=4):
    """
    preds, labels: (N, H, W)
    returns: (N,) tensor with mIoU per sample
    absent classes are ignored per sample
    """
    device = "cuda"
    classes = torch.arange(num_classes, device=device).view(1, num_classes, 1, 1)

    pred_c = preds.unsqueeze(1) == classes     # (N, C, H, W)
    label_c = labels.unsqueeze(1) == classes   # (N, C, H, W)

    intersection = (pred_c & label_c).sum(dim=(2, 3)).float()  # (N, C)
    union = (pred_c | label_c).sum(dim=(2, 3)).float()         # (N, C)

    valid = union > 0
    iou = intersection / union.clamp(min=1)

    miou_per_sample = (iou * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
    return miou_per_sample


def compute_prediction_consistency_(pred_ref, pred_pert):
    same = (pred_ref == pred_pert).float().mean()
    return same.item()


import torch

def compute_stats_and_outliers(values):
    """
    values: (N,) tensor

    returns:
        mean
        std
        q1
        q3
        iqr
        lower_bound
        upper_bound
        outliers (tensor)
        outlier_indices
    """
    values = values.float()

    mean = values.mean()
    std = values.std()

    q1 = torch.quantile(values, 0.25)
    q3 = torch.quantile(values, 0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (values < lower_bound) | (values > upper_bound)
    outliers = values[outlier_mask]
    outlier_indices = torch.where(outlier_mask)[0]

    return {
        "mean": mean.item(),
        "std": std.item(),
        "q1": q1.item(),
        "q3": q3.item(),
        "iqr": iqr.item(),
        "lower_bound": lower_bound.item(),
        "upper_bound": upper_bound.item(),
        "outliers": outliers,
        "outlier_indices": outlier_indices
    }

def compute_std_dev(miou_per_sample):
    """
    miou_per_sample: tensor of shape (N,)
    returns: float
    """
    return miou_per_sample.float().std().item()

# def compute_std_dev():
#     pass

# def compute_icc(pred_ref, pred_pert):
#     x = pred_ref.flatten().float()
#     y = pred_pert.flatten().float()

#     mean_x = x.mean()
#     mean_y = y.mean()

#     var_x = ((x - mean_x) ** 2).mean()
#     var_y = ((y - mean_y) ** 2).mean()
#     cov_xy = ((x - mean_x) * (y - mean_y)).mean()

#     icc = (2 * cov_xy) / (var_x + var_y + 1e-8)
#     return icc.item()


# def compute_ece_from_preds_and_conf(preds, confidences, labels, num_bins=15):
#     bin_boundaries = torch.linspace(0, 1, num_bins + 1)

#     preds = preds.view(-1)
#     confidences = confidences.view(-1)
#     labels = labels.view(-1)

#     correct = (preds == labels).float()
#     N = len(confidences)

#     ece = 0.0

#     for i in range(len(bin_boundaries) - 1):
#         lower = bin_boundaries[i]
#         upper = bin_boundaries[i + 1]

#         in_bin = (confidences > lower) & (confidences <= upper)
#         bin_count = in_bin.sum().item()

#         if bin_count > 0:
#             acc = correct[in_bin].mean().item()
#             conf = confidences[in_bin].mean().item()

#             ece += (bin_count / N) * abs(acc - conf)

#     return ece

# def compute_confidence(confs):
#     """
#     confs: tensor of shape [N, H, W]

#     Returns:
#         scalar: average confidence over all samples
#     """
#     # Step 1: average over pixels per sample → [N]
#     per_sample_conf = confs.mean(dim=(1, 2))

#     # Step 2: average over samples → scalar
#     overall_conf = per_sample_conf.mean()

#     return overall_conf.item()