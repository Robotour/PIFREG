# GAN 判别器（项目扩展，非官方 VoxelMorph 核心）

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """GAN 判别器，用于判别 fixed 与 warped 图像差异。"""

    def __init__(self, in_channels=2, base_channels=16):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        fc_input_dim = base_channels * 4 * (512 // 4) * (512 // 4)
        self.fc = nn.Linear(fc_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fixed, warped):
        x = torch.cat([fixed, warped], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.sigmoid(x)
