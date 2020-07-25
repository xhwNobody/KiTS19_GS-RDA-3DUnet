import torch
import torch.nn as nn

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
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x_out = self.conv(x)
        
        return x_out

class down_2_2_1(nn.Module):
    def __init__(self):
        super(down_2_2_1, self).__init__()
        self.down = nn.Sequential(
            nn.AdaptiveMaxPool3d((80, 80, 80))
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
        self.encoder5 = double_conv(240, 320)
        self.down5 = down_2_2_2()

        self.bottleneck = double_conv(320, 320)

        self.up5 = up_2_2_2(320, 320)
        self.decoder5 = double_conv(320*2, 320)
        self.up4 = up_2_2_2(320, 240)
        self.decoder4 = double_conv(240*2, 240)
        self.up3 = up_2_2_2(240, 120)
        self.decoder3 = double_conv(120*2, 120)
        self.up2 = up_2_2_2(120,60)
        self.decoder2 = signal_conv(60*2, 60)
        self.up1 = up_2_2_1(60, 30)
        self.decoder1 = signal_conv(30*2, 30)

        self.outc = outconv(30, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = self.encoder1(x)
        x1_down = self.down1(x1)

        x2 = self.encoder2(x1_down)
        x2_down = self.down2(x2)

        x3 = self.encoder3(x2_down)
        x3_down = self.down3(x3)

        x4 = self.encoder4(x3_down)
        x4_down = self.down4(x4)

        x5 = self.encoder5(x4_down)
        x5_down = self.down5(x5)

        x_bottleneck = self.bottleneck(x5_down)

        x5_up = self.up5(x_bottleneck)
        x5_ = torch.cat([x5, x5_up], dim=1)
        x5_ = self.decoder5(x5_)

        x4_up = self.up4(x5_)
        x4_ = torch.cat([x4, x4_up], dim=1)
        x4_ = self.decoder4(x4_)

        x3_up = self.up3(x4_)
        x3_ = torch.cat([x3, x3_up], dim=1)
        x3_ = self.decoder3(x3_)

        x2_up = self.up2(x3_)
        x2_ = torch.cat([x2, x2_up], dim=1)
        x2_ = self.decoder2(x2_)

        x1_up = self.up1(x2_)
        x1_ = torch.cat([x1, x1_up], dim=1)
        x1_ = self.decoder1(x1_)

        x_out = self.outc(x1_)

        return self.sigmoid(x_out)
