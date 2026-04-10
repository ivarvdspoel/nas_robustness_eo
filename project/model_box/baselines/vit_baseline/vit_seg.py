import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ViTSegmentation(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 256,
        in_chans: int = 7,
        pretrained: bool = False,
        backbone_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        self.patch_size = 16
        self.img_size = img_size

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
            dynamic_img_size=True,
        )

        self.embed_dim = self.backbone.num_features

        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input size must be divisible by patch size {self.patch_size}, got {(h, w)}"
            )

        feat_h = h // self.patch_size
        feat_w = w // self.patch_size

        tokens = self.backbone.forward_features(x)

        if tokens.ndim != 3:
            raise ValueError(f"Expected [B, N, C] tokens, got {tokens.shape}")

        expected_tokens = feat_h * feat_w

        if tokens.shape[1] == expected_tokens + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Unexpected number of tokens: got {tokens.shape[1]}, "
                f"expected {expected_tokens} or {expected_tokens + 1}"
            )

        feats = tokens.transpose(1, 2).reshape(b, self.embed_dim, feat_h, feat_w)
        logits = self.decoder(feats)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return logits