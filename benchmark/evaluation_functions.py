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

