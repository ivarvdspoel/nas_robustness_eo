# File with robustness evaluation metrics found in the literature
import torch

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