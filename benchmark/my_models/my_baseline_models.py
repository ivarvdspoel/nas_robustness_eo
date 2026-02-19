import torch
import torch.nn as nn
from torchvision import models
import timm

from my_models.my_unet import UNetDecoder

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()

        self.model = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4)  # 5 stages
        )

    def forward(self, x):
        features = self.model(x)
        return features
    
import torch
import torch.nn as nn
from torchvision import models
from my_models.my_unet import UNetDecoder


class EfficientNetB0Encoder(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()

        # features_only=True makes timm return multi-scale feature maps
        self.model = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4)  # 5 resolution stages
        )

    def forward(self, x):
        features = self.model(x)
        return features



class MobileOneS0Encoder(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()

        self.model = timm.create_model(
            "mobileone_s0",
            pretrained=True,
            features_only=True,
            in_chans=in_channels
        )

    def forward(self, x):
        features = self.model(x)
        return features  # already returns 5 feature maps


class ResNet18UNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=4):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=in_channels)
        
        input_res = (256, 256)
        dummy_input = torch.randn(1, in_channels, *input_res)
        
        with torch.no_grad():
            features = self.encoder(dummy_input)
            feature_shapes = [f.shape for f in features]
        
        # 2. Pass output_shape=input_res to the decoder
        # This tells the final Upsample layer exactly what size to target
        self.decoder = UNetDecoder(
            encoder_shapes=feature_shapes, 
            num_classes=num_classes,
            output_shape=input_res 
        )

    def forward(self, x):
        features = self.encoder(x) 
        logits = self.decoder(features)
        return logits
    


class MobileOneS0UNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=4):
        super().__init__()
        self.encoder = MobileOneS0Encoder(in_channels=in_channels)

        input_res = (256, 256)
        dummy_input = torch.randn(1, in_channels, *input_res)

        with torch.no_grad():
            features = self.encoder(dummy_input)
            feature_shapes = [f.shape for f in features]

        self.decoder = UNetDecoder(
            encoder_shapes=feature_shapes,
            num_classes=num_classes,
            output_shape=input_res,
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits


class EfficientNetB0UNet(nn.Module):
    def __init__(self, in_channels=7, num_classes=4):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(in_channels=in_channels)

        input_res = (256, 256)
        dummy_input = torch.randn(1, in_channels, *input_res)

        with torch.no_grad():
            features = self.encoder(dummy_input)
            feature_shapes = [f.shape for f in features]

        self.decoder = UNetDecoder(
            encoder_shapes=feature_shapes,
            num_classes=num_classes,
            output_shape=input_res,
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features)
        return logits
