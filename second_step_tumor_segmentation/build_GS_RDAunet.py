import torch
import torch.nn as nn
import torch.nn.functional as F
import GatedSpatialConv3d as gsc
from squeeze_and_excitation import ChannelSpatialSELayer


class signal_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(signal_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.csSE = ChannelSpatialSELayer(out_ch)
        self.conv11 = nn.Conv3d(in_ch, out_ch, 1)
        self.leakrelu = nn.LeakyReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
        )

    def forward(self, x):
        x_conv = self.conv(x)
        x_conv = self.csSE(x_conv)

        x_sti = self.conv11(x)

        x_mid = x_conv + x_sti
        x_out = self.leakrelu(x_mid)

        return x_out

class down_2_2_1(nn.Module):
    def __init__(self):
        super(down_2_2_1, self).__init__()
        self.down = nn.Sequential(
            nn.AdaptiveMaxPool3d((48, 64, 64))
        )

    def forward(self, x):
        x = self.down(x)
        return x

class down_2_2_2(nn.Module):
    def __init__(self):
        super(down_2_2_2, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1))
        )

    def forward(self, x):
        x = self.down(x)
        return x

class up_2_2_2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_2_2_2, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class up_2_2_1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_2_2_1, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0,1,1))

    def forward(self, x):
        x = self.up(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class gateSpatialConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(gateSpatialConv3d, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.InstanceNorm3d(in_ch+1),
            nn.Conv3d(in_ch+1, in_ch+1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(in_ch+1, 1, 1),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.outconv = nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, input_features, gating_features):
        alphas = self.gate_conv(torch.cat((input_features, gating_features), dim=1))
        return alphas

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.encoder1 = signal_conv(n_channels, 30)
        self.down1 = down_2_2_1()
        self.encoder2 = signal_conv(30 ,60)
        self.down2 = down_2_2_2()
        self.encoder3 = double_conv(60, 120)
        self.down3 = down_2_2_2()
        self.encoder4 = double_conv(120, 240)
        self.down4 = down_2_2_2()
        self.bottleneck = double_conv(240, 320)
        self.up4 = up_2_2_2(320, 240)
        self.decoder4 = double_conv(240*2, 240)
        self.up3 = up_2_2_2(240, 120)
        self.decoder3 = double_conv(120*2, 120)
        self.up2 = up_2_2_2(120,60)
        self.decoder2 = signal_conv(60*2, 60)
        self.up1 = up_2_2_1(60, 30)
        self.decoder1 = signal_conv(30*2, 30)
        self.outc = outconv(30, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self.interpolate = F.interpolate
        self.dsn2 = nn.Conv3d(120, 1, 1)
        self.dsn3 = nn.Conv3d(60, 1, 1)
        self.dsn4 = nn.Conv3d(30, 1, 1)

        self.res1 = double_conv(240*2, 64)
        self.d1 = nn.Conv3d(64, 32, 1)
        self.res2 = double_conv(32, 32)
        self.d2 = nn.Conv3d(32, 16, 1)
        self.res3 = double_conv(16, 16)
        self.d3 = nn.Conv3d(16, 8, 1)
        self.fuse = nn.Conv3d(8, 1, 1)

        self.gate1 = gateSpatialConv3d(32, 32)
        self.gate2 = gateSpatialConv3d(16, 16)
        self.gate3 = gateSpatialConv3d(8, 8)
        self.sigmoid = nn.Sigmoid()

        self.conv_edge = nn.Conv3d(8, 30, 1)
        self.final_fuse = signal_conv(60, 30)


    def forward(self, x):
        x_size = x.size()

        #Regular Stream
        x1 = self.encoder1(x)
        x1_down = self.down1(x1)
        x2 = self.encoder2(x1_down)
        x2_down = self.down2(x2)
        x3 = self.encoder3(x2_down)
        x3_down = self.down3(x3)
        x4 = self.encoder4(x3_down)
        x4_down = self.down4(x4)
        x_bottleneck = self.bottleneck(x4_down)
        x4_up = self.up4(x_bottleneck)

        x4_cat = torch.cat((x4, x4_up), dim=1)
        x4_ = self.decoder4(x4_cat)
        x3_up = self.up3(x4_)
        x3_cat = torch.cat((x3, x3_up), dim=1)
        x3_ = self.decoder3(x3_cat)
        x2_up = self.up2(x3_)
        x2_cat = torch.cat((x2, x2_up), dim=1)
        x2_ = self.decoder2(x2_cat)
        x1_up = self.up1(x2_)
        x1_cat = torch.cat((x1, x1_up), dim=1)
        x1_ = self.decoder1(x1_cat)

        #Shape Stream
        cs = self.res1(x4_cat)
        cs = self.d1(cs)
        cs = self.interpolate(cs, x_size[2:], mode='trilinear', align_corners=True)
        s2 = self.interpolate(self.dsn2(x3_), x_size[2:], mode='trilinear', align_corners=True)
        alpha1 = self.gate1(cs, s2)
        cs = (cs * (alpha1 + 1))

        cs = self.res2(cs)
        cs = self.d2(cs)
        s3 = self.interpolate(self.dsn3(x2_), x_size[2:], mode='trilinear', align_corners=True)
        alpha2 = self.gate2(cs, s3)
        cs = (cs * (alpha2 + 1))

        cs = self.res3(cs)
        cs = self.d3(cs)
        s4 = self.interpolate(self.dsn4(x1_), x_size[2:], mode='trilinear', align_corners=True)
        alpha3 = self.gate3(cs, s4)
        cs = (cs * (alpha3 + 1))

        cs_fuse= self.fuse(cs)
        final_edge = self.sigmoid(cs_fuse)

        #Fusion Module
        final_edge_x = self.conv_edge(cs)#up dims
        x_fuse = torch.cat((final_edge_x, x1_), dim=1)
        x_fuse = self.final_fuse(x_fuse)
        x_out = self.outc(x_fuse)
        final_sigment = self.softmax(x_out)

        return final_sigment, final_edge
