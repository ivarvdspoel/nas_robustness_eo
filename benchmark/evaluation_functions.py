# File with robustness evaluation metrics found in the literature
import torch



def compute_my_miou_from_preds(preds, labels, num_classes=4):
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
    """
    N = preds.shape[0]
    miou_per_sample = torch.zeros(N)

    for i in range(N):
        ious = []
        for cls in range(num_classes):
            pred_mask = (preds[i] == cls)
            label_mask = (labels[i] == cls)

            union = (pred_mask | label_mask).sum()
            if union == 0:
                continue  # ignore absent classes
            inter = (pred_mask & label_mask).sum()
            ious.append(inter.float() / union.float())

        if len(ious) == 0:
            miou_per_sample[i] = 0.0
        else:
            miou_per_sample[i] = torch.stack(ious).mean()

    return miou_per_sample

def compute_miou(model, dataloader, num_classes=4, device="cpu"):
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            # Convert one-hot to class indices if needed
            if y.ndim == 4:
                y = torch.argmax(y, dim=1)
            y = y.long().to(device)

            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            for cls in range(num_classes):
                inter = ((preds == cls) & (y == cls)).sum()
                uni = ((preds == cls) | (y == cls)).sum()
                intersection[cls] += inter
                union[cls] += uni

    return (intersection / union).mean().item()


def prediction_consistency(pred_ref, pred_pert):
    same = (pred_ref == pred_pert).float().mean()
    return same.item()


def compute_icc(pred_ref, pred_pert):
    x = pred_ref.flatten().float()
    y = pred_pert.flatten().float()

    mean_x = x.mean()
    mean_y = y.mean()

    var_x = ((x - mean_x) ** 2).mean()
    var_y = ((y - mean_y) ** 2).mean()
    cov_xy = ((x - mean_x) * (y - mean_y)).mean()

    icc = (2 * cov_xy) / (var_x + var_y + 1e-8)
    return icc.item()


def compute_ece_from_preds_and_conf(preds, confidences, labels, num_bins=15):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)

    preds = preds.view(-1)
    confidences = confidences.view(-1)
    labels = labels.view(-1)

    correct = (preds == labels).float()
    N = len(confidences)

    ece = 0.0

    for i in range(len(bin_boundaries) - 1):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = (confidences > lower) & (confidences <= upper)
        bin_count = in_bin.sum().item()

        if bin_count > 0:
            acc = correct[in_bin].mean().item()
            conf = confidences[in_bin].mean().item()

            ece += (bin_count / N) * abs(acc - conf)

    return ece

def compute_confidence(confs):
    """
    confs: tensor of shape [N, H, W]

    Returns:
        scalar: average confidence over all samples
    """
    # Step 1: average over pixels per sample → [N]
    per_sample_conf = confs.mean(dim=(1, 2))

    # Step 2: average over samples → scalar
    overall_conf = per_sample_conf.mean()

    return overall_conf.item()