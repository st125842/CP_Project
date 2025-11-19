import torch
import torch.nn as nn

# -------------------------
# Swin Transformer Block 2D
# -------------------------
class SwinBlock2D(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.window_size = window_size

    def window_partition(self, x):
        B, C, H, W = x.shape
        w = self.window_size
        x = x.view(B, C, H // w, w, W // w, w)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, w * w, C)  # (num_windows*B, w*w, C)
        return x, (H, W)

    def window_reverse(self, x, H, W):
        B = x.shape[0] // ((H // self.window_size) * (W // self.window_size))
        w = self.window_size
        x = x.view(B, H // w, W // w, w, w, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        x_windows, (H_ori, W_ori) = self.window_partition(x)

        x_windows = self.norm1(x_windows)
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        x_windows = x_windows + attn_out
        x_windows = x_windows + self.mlp(self.norm2(x_windows))

        x = self.window_reverse(x_windows, H_ori, W_ori)
        return x

# -------------------------
# SwinUNet2D (encoder-decoder)
# -------------------------
class SwinUNet2D(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_dim=64):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_dim, 3, padding=1),
            nn.ReLU(True)
        )
        self.swin1 = SwinBlock2D(base_dim, num_heads=8)

        self.down1 = nn.Conv2d(base_dim, base_dim * 2, 4, 2, 1)
        self.swin2 = SwinBlock2D(base_dim * 2, num_heads=8)

        # Bottleneck
        self.down2 = nn.Conv2d(base_dim * 2, base_dim * 4, 4, 2, 1)
        self.swin3 = SwinBlock2D(base_dim * 4, num_heads=8)

        # Decoder
        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, 2, 1)
        self.swin4 = SwinBlock2D(base_dim * 2, num_heads=8)

        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, 2, 1)
        self.swin5 = SwinBlock2D(base_dim, num_heads=8)

        self.final = nn.Sequential(
            nn.Conv2d(base_dim, out_ch, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x1 = self.swin1(x1)

        x2 = self.down1(x1)
        x2 = self.swin2(x2)

        x3 = self.down2(x2)
        x3 = self.swin3(x3)

        x4 = self.up1(x3)
        x4 = self.swin4(x4 + x2)  # skip connection

        x5 = self.up2(x4)
        x5 = self.swin5(x5 + x1)  # skip connection

        out = self.final(x5)
        return out

# -------------------------
# Test with CUDA
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = SwinUNet2D(in_ch=3, out_ch=1).to(device)
x = torch.randn(1, 3, 256, 256).to(device)
y = model(x)

print("Input:", x.shape)
print("Output:", y.shape)
