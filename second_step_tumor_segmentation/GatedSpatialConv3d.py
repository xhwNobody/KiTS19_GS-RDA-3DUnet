import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _triple

class CatedSpatialConv3d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(CatedSpatialConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

        self._gate_conv = nn.Sequential(
            nn.InstanceNorm3d(in_channels+1),
            nn.Conv3d(in_channels+1, in_channels+1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(in_channels+1, 1, 1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
    def forward(self, input_features, gating_features):
        alphas = self._gate_conv(torch.cat((input_features, gating_features), dim=1))
        input_features = (input_features * (alphas + 1))
        return F.conv3d(input_features, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
