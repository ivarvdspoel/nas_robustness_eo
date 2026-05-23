import torch



def compute_miou(preds, labels, num_classes=4):
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    for cls in range(num_classes):
        inter = ((preds == cls) & (labels == cls)).sum()
        uni = ((preds == cls) | (labels == cls)).sum()
        intersection[cls] += inter
        union[cls] += uni

    return (intersection / union).mean().item()


def compute_prediction_consistency(pred_ref, pred_pert):
    same = (pred_ref == pred_pert).float().mean()
    return same.item()

