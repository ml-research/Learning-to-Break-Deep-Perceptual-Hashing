import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math


def ssim_loss(orig_image, optim_image, window_size=11, channel=3):
    # Source: https://github.com/Po-Hsun-Su/pytorch-ssim
    if torch.all(torch.eq(orig_image, optim_image)):
        return torch.tensor(0.0)

    def gaussian(window_size, sigma):
        gauss = torch.Tensor(
            [math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(
        channel, 1, window_size, window_size).contiguous()).to(orig_image.device)

    mu1 = F.conv2d(orig_image, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(optim_image, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(
        orig_image*orig_image, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(
        optim_image*optim_image, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(orig_image*optim_image, window, padding=window_size //
                       2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
