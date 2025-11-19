import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

# ---------- SSIM utility ----------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    if val_range is None:
        max_val = 1 if torch.max(img1) <= 1 else 255
        min_val = 0
    else:
        min_val, max_val = val_range

    L = max_val - min_val  # dynamic range
    padd = window_size // 2

    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# ---------- Combined MAE + SSIM loss ----------
class MAE_SSIM_Loss(nn.Module):
    def __init__(self, alpha=0.84):  # alpha = weight of SSIM (typical 0.8~0.9)
        super(MAE_SSIM_Loss, self).__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        mae = self.l1(pred, target)
        ssim_val = ssim(pred, target)
        loss = (1 - self.alpha) * mae + self.alpha * (1 - ssim_val)
        return loss

# ---------- Example ----------
if __name__ == "__main__":
    pred = torch.rand(1, 1, 256, 256).cuda()
    target = torch.rand(1, 1, 256, 256).cuda()
    loss_fn = MAE_SSIM_Loss(alpha=0.84).cuda()
    loss = loss_fn(pred, target)
    print("Loss:", loss.item())
