import torch
import torch.nn as nn

class UNetDecoder(nn.Module):
    """
    A PyTorch implementation of a U-Net decoder module.
    This class implements the decoder part of the U-Net architecture, which
    reconstructs the output from the bottleneck features by progressively
    upsampling and combining them with skip connections from the encoder.
    Attributes:
        num_stages (int): The number of decoding stages, equal to the number of skip connections.
        up_convs (nn.ModuleList): A list of transposed convolution layers for upsampling.
        conv_blocks (nn.ModuleList): A list of convolutional blocks for processing concatenated
                                     upsampled and skip connection features.
        out_conv (nn.Conv2d): The final convolutional layer that produces the output.
    Args:
        encoder_shapes (list of torch.Size): A list of shapes of the encoder features in the order:
                                             [skip0, skip1, ..., skip_(N-1), bottleneck].
                                             Each shape is expected to be a `torch.Size` object.
        num_classes (int, optional): The number of output classes. Default is 2.
    Methods:
        forward(encoder_features):
            Performs the forward pass of the decoder.
            Args:
                encoder_features (list of torch.Tensor): A list of encoder feature maps in the order:
                                                         [skip0, skip1, ..., skip_(N-1), bottleneck].
                                                         The number of feature maps must match the
                                                         number of stages + 1.
            Returns:
                torch.Tensor: The output tensor after decoding.
    """
    def __init__(self, encoder_shapes, num_classes=2, output_shape=None):
        super(UNetDecoder, self).__init__()
        # Expecting encoder_shapes as a list of torch.Size objects in order:
        # [skip0, skip1, ..., skip_(N-1), bottleneck]
        # Number of decoding stages equals the number of skip connections.
        self.num_stages = len(encoder_shapes) - 1
        
        # Build upsampling and convolution blocks dynamically.
        self.up_convs = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        
        # Initial number of channels from the bottleneck.
        in_channels = encoder_shapes[-1][1]
        
        # Iterate over skip connections in reverse order.
        for i in range(self.num_stages - 1, -1, -1):
            skip_channels = encoder_shapes[i][1]
            # Store the expected output size for each upsampling operation
            out_h, out_w = encoder_shapes[i][2], encoder_shapes[i][3]
            
            # Replace transposed convolution with upsampling + conv
            self.up_convs.append(
            nn.Sequential(
                nn.Upsample(size=(out_h, out_w), mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, skip_channels, kernel_size=1)
            )
            )
            
            # The conv block takes the concatenated tensor (upsampled + skip) as input.
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(skip_channels, skip_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(skip_channels),
                    nn.ReLU(inplace=True)
                )
            )
            # Update in_channels to be the skip_channels for the next stage.
            in_channels = skip_channels

        # Final output convolution followed by upsampling to match the input height and width.
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
            nn.Upsample(size=output_shape, mode='bilinear', align_corners=False)
        )

    def forward(self, encoder_features, verbose=False):
        # encoder_features: list in order [skip0, skip1, ..., skip_(N-1), bottleneck]
        if verbose:
            print("=" * 100)
            print(f"{'Encoder Feature Shapes':^100}")
            print(encoder_features.shape)
            print("=" * 100)

        if len(encoder_features) != self.num_stages + 1:
            raise ValueError(f"Expected {self.num_stages + 1} encoder features, but got {len(encoder_features)}.")

        x = encoder_features[-1]  # start with bottleneck

        # For each decoding stage, use the corresponding skip connection in reverse order.
        for i in range(self.num_stages):
            # Skip connection index: from last skip to the first.
            skip = encoder_features[self.num_stages - 1 - i]
            x = self.up_convs[i](x)
            # Match spatial dimensions with skip connection (just in case upsampling mismatch occurs)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.conv_blocks[i](x)

        output = self.out_conv(x)
        return output