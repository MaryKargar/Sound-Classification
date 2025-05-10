import torch
import torch.nn as nn
import random

# SpecAugment block: applied only during training
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=25, num_masks=2):
        super(SpecAugment, self).__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, x):
        for _ in range(self.num_masks):
            freq_mask = random.randint(0, self.freq_mask_param)
            freq_start = random.randint(0, max(0, x.size(2) - freq_mask))
            x[:, :, freq_start:freq_start + freq_mask, :] = 0

            time_mask = random.randint(0, self.time_mask_param)
            time_start = random.randint(0, max(0, x.size(3) - time_mask))
            x[:, :, :, time_start:time_start + time_mask] = 0
        return x

# SEBlock for channel-wise attention
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ResNet-like basic block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# Main audio classification model
class AudioMLP(nn.Module):
    def __init__(self, num_classes=50, dropout_p=0.1, use_attention=True):
        super(AudioMLP, self).__init__()
        self.in_channels = 128
        self.specaugment = SpecAugment()
        self.dropout = nn.Dropout(p=dropout_p)

        self.conv1 = nn.Conv2d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 3, 128, stride=1)
        self.se1 = SEBlock(128) if use_attention else nn.Identity()

        self.layer2 = self._make_layer(BasicBlock, 3, 256, stride=2)
        self.se2 = SEBlock(256) if use_attention else nn.Identity()

        self.layer3 = self._make_layer(BasicBlock, 3, 512, stride=2)
        self.se3 = SEBlock(512) if use_attention else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, blocks, out_channels, stride):
        downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            x = self.specaugment(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.se2(x)
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.se3(x)
        x = self.dropout(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
