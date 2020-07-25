from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.fc1 = nn.Conv3d(num_channels, num_channels_reduced, kernel_size=1, padding=0)
        self.fc2 = nn.Conv3d(num_channels_reduced, num_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, pooling = False):
        module_input = input_tensor
        x = self.avg_pool(input_tensor)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class SpatialSELayer(nn.Module):

    def __init__(self, num_channels):
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        mid = self.conv(input_tensor)
        mid = self.sigmoid(mid)
        return input_tensor * mid


class ChannelSpatialSELayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        output_tensor = torch.add(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor



