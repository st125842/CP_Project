import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Basic Building Blocks
# -----------------------

class ConvBlock(nn.Module):
    """Two convolutional layers with residual connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual connection (1x1 conv if dimensions differ)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + res)
        return x


class EncoderBlock(nn.Module):
    """Downsampling block: ConvBlock + MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    """Upsampling block: ConvTranspose + ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        # Align spatial dimensions (in case of mismatch)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# -----------------------
# ResUNet Model
# -----------------------

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for feature in features:
            in_c = in_channels if len(self.encoder_blocks) == 0 else features[len(self.encoder_blocks) - 1]
            self.encoder_blocks.append(EncoderBlock(in_c, feature))

        # Bridge
        self.bridge = ConvBlock(features[-1], features[-1] * 2)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        reversed_features = features[::-1]
        for feature in reversed_features:
            in_c = (feature * 2) if feature != reversed_features[0] else features[-1] * 2
            self.decoder_blocks.append(DecoderBlock(in_c, feature))

        # Final output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoder_blocks:
            x, x_pooled = enc(x)
            skips.append(x)
            x = x_pooled

        x = self.bridge(x)
        skips = skips[::-1]

        for idx, dec in enumerate(self.decoder_blocks):
            x = dec(x, skips[idx])

        return torch.sigmoid(self.final_conv(x))


# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    model = Generator(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)  # → torch.Size([1, 1, 256, 256])
