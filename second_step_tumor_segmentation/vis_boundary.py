import sys
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from torch import optim
from optparse import OptionParser
from build_GS_RDAunet import UNet
from PIL import Image
import skimage.io as io
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
import nibabel as nib



def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))



def UNet_Mid(net, x):

    x_size = x.size()
    #Regular Stream
    x1 = net.encoder1(x)
    x1_down = net.down1(x1)
    x2 = net.encoder2(x1_down)
    x2_down = net.down2(x2)
    x3 = net.encoder3(x2_down)
    x3_down = net.down3(x3)
    x4 = net.encoder4(x3_down)
    x4_down = net.down4(x4)
    x_bottleneck = net.bottleneck(x4_down)
    x4_up = net.up4(x_bottleneck)
    x4_ = torch.cat((x4, x4_up), dim=1)
    x4_ = net.decoder4(x4_)
    x3_up = net.up3(x4_)
    x3_ = torch.cat((x3, x3_up), dim=1)
    x3_ = net.decoder3(x3_)
    x2_up = net.up2(x3_)
    x2_ = torch.cat((x2, x2_up), dim=1)
    x2_ = net.decoder2(x2_)
    x1_up = net.up1(x2_)
    x1_ = torch.cat((x1, x1_up), dim=1)
    x1_ = net.decoder1(x1_)

    #Shape Stream
    cs = net.res1(x1)
    cs = net.d1(cs)
    s2 = net.interpolate(net.dsn2(x2), x_size[2:], mode='trilinear', align_corners=True)
    cs = net.gate1(cs, s2)

    cs = net.res2(cs)
    cs = net.d2(cs)
    s3 = net.interpolate(net.dsn3(x3), x_size[2:], mode='trilinear', align_corners=True)
    cs = net.gate2(cs, s3)

    cs = net.res3(cs)
    cs = net.d3(cs)
    s4 = net.interpolate(net.dsn4(x4), x_size[2:], mode='trilinear', align_corners=True)
    cs = net.gate3(cs, s4)

    cs = net.fuse(cs)
    final_edge = net.sigmoid(cs)
    return final_edge


if __name__ == "__main__":
    img = nib.load('/home/xhw/桌面/kits19-tumor-GS-RDAUnet/val_Image/004_right_000_img.nii.gz')
    img_org_data = img.get_data()
    img_org_affine = img.affine
    img_data = np.expand_dims(img_org_data, -1)
    img_data = np.swapaxes(np.swapaxes(img_data, 0, 3), 1, 2)
    img_data = np.expand_dims(img_data, 0)


    net = UNet(n_channels=1, n_classes=3)

    model = '/home/xhw/桌面/kits19-tumor-GS-RDAUnet/checkpoints/CP60_val_0.813813.pth'

    # net.cuda()
    net.load_state_dict(torch.load(model))
    #print(net)

    img_data = torch.from_numpy(img_data).type(torch.FloatTensor)
    result = UNet_Mid(net, img_data).cpu().detach().numpy()
    #
    # result = result.detach().numpy()
    result = np.squeeze(result)
    result = np.swapaxes(result, 0, 2)
    result = (result >= 0.5)+0.

    print(result.shape)
    print(np.sum(result))

    mskdata = nib.Nifti1Image(result, img_org_affine)
    nib.save(mskdata, '004_right_000_boundary_pre.nii.gz')



