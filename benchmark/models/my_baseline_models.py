import torch
import torch.nn as nn
from torchvision import models


from models.my_unet import UNetDecoder

class ResNet18Encoder(nn.Module):
    def __init__(self, in_channels=7):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 7-channel input adaptation
        self.init_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.init_conv.weight[:, :3] = base.conv1.weight
            self.init_conv.weight[:, 3:] = base.conv1.weight.mean(dim=1, keepdim=True)

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        
        self.layer1 = base.layer1  
        self.layer2 = base.layer2  
        self.layer3 = base.layer3  
        self.layer4 = base.layer4  

    def forward(self, x):
        # We need to collect "skips" for the UNetDecoder
        x = self.init_conv(x)
        x = self.bn1(x)
        x0 = self.relu(x)      # Skip 0: (64, 128, 128)
        
        x_m = self.maxpool(x0)
        x1 = self.layer1(x_m)  # Skip 1: (64, 64, 64)
        x2 = self.layer2(x1)   # Skip 2: (128, 32, 32)
        x3 = self.layer3(x2)   # Skip 3: (256, 16, 16)
        x4 = self.layer4(x3)   # Bottleneck: (512, 8, 8)
        
        return [x0, x1, x2, x3, x4]
    
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