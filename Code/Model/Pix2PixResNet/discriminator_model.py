import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Residual Block
# -----------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out = F.leaky_relu(out + identity, 0.2)
        return out


# -----------------------
# ResUNet Encoder Discriminator
# -----------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        # Initial Layer (like in PatchGAN)
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        # Residual Encoder Layers
        layers = []
        in_c = features[0]
        for i, feature in enumerate(features[1:]):
            stride = 1 if feature == features[-1] else 2  # last layer: no downsample
            layers.append(ResidualBlock(in_c, feature, stride=stride))
            in_c = feature

        # Final output layer (PatchGAN prediction map)
        layers.append(
            nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # Concatenate input and target along channel dim
        # print(x.shape)
        # print(y.shape)
        # print('----------')
        x = torch.cat([x, y], dim=1)
        # print(x.shape)
        # print('----------')
        x = self.initial(x)
        x = self.model(x)
        x = F.interpolate(x, size=(30, 30), mode="bilinear", align_corners=False)

        return x


# -----------------------
# Test it
# -----------------------
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(model)
    print("Output shape:", preds.shape)  # typically [1, 1, 30, 30] for PatchGAN


if __name__ == "__main__":
    test()
