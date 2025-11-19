import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Utilities ----------------
def conv3x3(in_ch, out_ch):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

def norm_act(ch):
    # GroupNorm with sensible number of groups (min(8, ch))
    groups = min(8, ch) if ch >= 2 else 1
    return nn.Sequential(nn.GroupNorm(groups, ch), nn.GELU())

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch),
            norm_act(out_ch),
            conv3x3(out_ch, out_ch),
            norm_act(out_ch)
        )
    def forward(self, x):
        return self.block(x)

# ---------------- CBAM Module ----------------
class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        hidden = max(in_ch // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, in_ch, bias=False)
        )
    def forward(self, x):
        b, c, h, w = x.shape
        avg = torch.mean(x.view(b, c, -1), dim=2)            # (b,c)
        max_ = torch.max(x.view(b, c, -1), dim=2)[0]         # (b,c)
        out = self.mlp(avg) + self.mlp(max_)
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True)[0]
        cat = torch.cat([avg, max_], dim=1)
        scale = torch.sigmoid(self.conv(cat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_ch, reduction)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ---------------- Self-Attention (SAGAN) ----------------
class SelfAttention2d(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.key   = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        b, c, h, w = x.size()
        q = self.query(x).view(b, -1, h*w)           # (b, c//8, N)
        k = self.key(x).view(b, -1, h*w)             # (b, c//8, N)
        v = self.value(x).view(b, -1, h*w)           # (b, c, N)
        attn = torch.bmm(q.permute(0,2,1), k)        # (b, N, N)
        attn = F.softmax(attn / (q.shape[1]**0.5), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))      # (b, c, N)
        out = out.view(b, c, h, w)
        return self.gamma * out + x

# ---------------- Transformer Block ----------------
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        # x: (B, N, C)
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + attn_out
        x2 = self.norm2(x)
        x = x + self.mlp(x2)
        return x

# ---------------- Cross-Attention Block ----------------
class CrossAttention2d(nn.Module):
    def __init__(self, query_ch, key_ch, num_heads=8):
        super().__init__()
        # We'll flatten spatial dims and use nn.MultiheadAttention
        self.query_proj = nn.Conv2d(query_ch, query_ch, 1)
        self.key_proj   = nn.Conv2d(key_ch, query_ch, 1)   # map keys to query dim
        self.value_proj = nn.Conv2d(key_ch, query_ch, 1)
        self.mha = nn.MultiheadAttention(embed_dim=query_ch, num_heads=num_heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(query_ch), nn.Linear(query_ch, query_ch), nn.GELU(), nn.Linear(query_ch, query_ch))
    def forward(self, query_feat, key_feat):
        # query_feat: (b, Cq, Hq, Wq)
        # key_feat:   (b, Ck, Hk, Wk)
        b, Cq, Hq, Wq = query_feat.shape
        _, Ck, Hk, Wk = key_feat.shape
        q = self.query_proj(query_feat).flatten(2).permute(0,2,1)  # (b, Nq, Cq)
        k = self.key_proj(key_feat).flatten(2).permute(0,2,1)      # (b, Nk, Cq)
        v = self.value_proj(key_feat).flatten(2).permute(0,2,1)    # (b, Nk, Cq)
        out, _ = self.mha(q, k, v)  # (b, Nq, Cq)
        out = out + q
        out = out + self.ff(out)
        out = out.permute(0,2,1).view(b, Cq, Hq, Wq)
        return out

# ---------------- Base UNet skeleton ----------------
class UNetBase(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], final_act='tanh'):
        super().__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.downs = nn.ModuleList([ConvBlock(in_ch if i==0 else features[i-1], features[i]) for i in range(len(features))])
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)
        self.up_transpose = nn.ModuleList([nn.ConvTranspose2d(features[i]*2, features[i], kernel_size=2, stride=2) for i in reversed(range(len(features)))])
        self.up_conv = nn.ModuleList([ConvBlock(features[i]*2, features[i]) for i in reversed(range(len(features)))])
        self.final = nn.Conv2d(features[0], out_ch, 1)
        if final_act == 'tanh':
            self.final_act = nn.Tanh()
        elif final_act == 'sigmoid':
            self.final_act = nn.Sigmoid()
        else:
            self.final_act = nn.Identity()

    def encode(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips

    def decode(self, x, skips):
        skips = skips[::-1]
        for i in range(len(self.up_transpose)):
            x = self.up_transpose[i](x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.up_conv[i](x)
        return x

    def forward(self, x):
        raise NotImplementedError

# ---------------- Variant 1: CBAM Attention UNet ----------------
class CBAM_UNet(UNetBase):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__(in_ch, out_ch, features)
        # attach CBAM modules on each skip
        self.cbams = nn.ModuleList([CBAM(f) for f in features])

    def forward(self, x):
        x, skips = self.encode(x)
        # apply cbam to skips
        skips = [self.cbams[i](skips[i]) for i in range(len(skips))]
        x = self.bottleneck(x)
        x = self.decode(x, skips)
        x = self.final(x)
        return self.final_act(x)

# ---------------- Variant 2: Self-Attention UNet ----------------
class SelfAttUNet(UNetBase):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512]):
        super().__init__(in_ch, out_ch, features)
        # self-attention in bottleneck + at decoder levels optionally
        self.sa_bottleneck = SelfAttention2d(features[-1]*2)
        # optional: self-attention at decoder last two levels
        self.sa_dec1 = SelfAttention2d(features[-1])
        self.sa_dec2 = SelfAttention2d(features[-2])

    def forward(self, x):
        x, skips = self.encode(x)
        x = self.bottleneck(x)
        x = self.sa_bottleneck(x)
        # decode but inject extra self-attention after concat convs
        skips = skips
        x = self.decode(x, skips)
        # apply small SA on final decoder feature maps (optional)
        # safe-check shapes:
        if hasattr(self, 'sa_dec1') and x.shape[1] >= self.sa_dec1.query.in_channels:
            try:
                x = self.sa_dec1(x)
            except Exception:
                pass
        x = self.final(x)
        return self.final_act(x)

# ---------------- Variant 3: Transformer UNet (bottleneck transformer) ----------------
class TransUNet(UNetBase):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], trans_dim=512, num_heads=8):
        super().__init__(in_ch, out_ch, features)
        # project bottleneck to trans_dim
        self.bottleneck_proj = nn.Conv2d(features[-1]*2, trans_dim, 1)
        self.inverse_proj = nn.Conv2d(trans_dim, features[-1]*2, 1)
        self.transformer = SimpleTransformerBlock(dim=trans_dim, num_heads=num_heads)
    def forward(self, x):
        x, skips = self.encode(x)
        x = self.bottleneck(x)                        # (b, Cb, H, W)
        b, Cb, H, W = x.shape
        x = self.bottleneck_proj(x)                   # (b, trans_dim, H, W)
        x = x.flatten(2).permute(0,2,1)               # (b, N, C)
        x = self.transformer(x)                       # (b, N, C)
        x = x.permute(0,2,1).view(b, -1, H, W)        # (b, trans_dim, H, W)
        x = self.inverse_proj(x)                      # map back
        x = self.decode(x, skips)
        x = self.final(x)
        return self.final_act(x)

# ---------------- Variant 4: Cross-Attention UNet ----------------
class CrossAttUNet(UNetBase):
    def __init__(self, in_ch=1, out_ch=1, features=[64,128,256,512], cross_heads=8):
        super().__init__(in_ch, out_ch, features)
        # We'll use cross-attention between decoder feature (query) and encoder skip (key/value)
        # create cross-att blocks for each decoder upsample stage (matching skip channels)
        self.cross_blocks = nn.ModuleList([CrossAttention2d(features[i]*2, features[i], num_heads=cross_heads) for i in reversed(range(len(features)))])
    def forward(self, x):
        x, skips = self.encode(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(len(self.up_transpose)):
            x = self.up_transpose[i](x)
            skip = skips[i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            # cross-attention: query is x (decoder), key/value is skip (encoder)
            x_att = self.cross_blocks[i](x, skip)
            # fuse: concat cross-attended decoder with skip
            concat = torch.cat([skip, x_att], dim=1)
            x = self.up_conv[i](concat)
        x = self.final(x)
        return self.final_act(x)

# ---------------- Example usage ----------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = torch.randn(1,1,256,256).to(device)

    print("CBAM_UNet ->", CBAM_UNet().to(device)(input).shape)
    print("SelfAttUNet ->", SelfAttUNet().to(device)(input).shape)
    print("TransUNet ->", TransUNet().to(device)(input).shape)
    # print("CrossAttUNet ->", CrossAttUNet().to(device)(input).shape)
