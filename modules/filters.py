import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gaussian_kernel(kernel_size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    return kernel / np.sum(kernel)


class GaussianBlur(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(GaussianBlur, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2  # Ensure output size matches input size
        self.register_buffer('kernel', torch.tensor(gaussian_kernel(kernel_size, sigma), dtype=torch.float32))
        self.kernel = self.kernel.view(1, 1, kernel_size, kernel_size)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)  # Repeat the kernel for each input channel

    def forward(self, x):
        x = F.conv2d(x, self.kernel.to(x), padding=self.padding, groups=self.channels)
        return x


gaussian_filter_2d = GaussianBlur(4, 7, 0.8)
