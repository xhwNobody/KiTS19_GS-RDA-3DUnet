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

    x4_cat = torch.cat((x4, x4_up), dim=1)
    x4_ = net.decoder4(x4_cat)
    x3_up = net.up3(x4_)
    x3_cat = torch.cat((x3, x3_up), dim=1)
    x3_ = net.decoder3(x3_cat)
    x2_up = net.up2(x3_)
    x2_cat = torch.cat((x2, x2_up), dim=1)
    x2_ = net.decoder2(x2_cat)
    x1_up = net.up1(x2_)
    x1_cat = torch.cat((x1, x1_up), dim=1)
    x1_ = net.decoder1(x1_cat)

    #Shape Stream
    cs = net.res1(x4_cat)
    cs = net.d1(cs)
    cs = net.interpolate(cs, x_size[2:], mode='trilinear', align_corners=True)
    s2 = net.interpolate(net.dsn2(x3_), x_size[2:], mode='trilinear', align_corners=True)
    alpha1 = net.gate1(cs, s2)
    cs = (cs * (alpha1 + 1))

    cs = net.res2(cs)
    cs = net.d2(cs)
    s3 = net.interpolate(net.dsn3(x2_), x_size[2:], mode='trilinear', align_corners=True)
    alpha2 = net.gate2(cs, s3)
    cs = (cs * (alpha2 + 1))

    cs = net.res3(cs)
    cs = net.d3(cs)
    s4 = net.interpolate(net.dsn4(x1_), x_size[2:], mode='trilinear', align_corners=True)
    alpha3 = net.gate3(cs, s4)
    cs = (cs * (alpha3 + 1))

    cs_fuse= net.fuse(cs)
    final_edge = net.sigmoid(cs_fuse)

    return alpha1, alpha2, alpha3, final_edge


if __name__ == "__main__":
    img = nib.load('/home/xhw/桌面/kits19-tumor-GS-RDAUnet/val_Image/004_right_000_img.nii.gz')
    img_org_data = img.get_data()
    img_org_affine = img.affine
    img_data = np.expand_dims(img_org_data, -1)
    img_data = np.swapaxes(np.swapaxes(img_data, 0, 3), 1, 2)
    img_data = np.expand_dims(img_data, 0)

    net = UNet(n_channels=1, n_classes=3)

    model = '/home/xhw/桌面/kits19-tumor-GS-RDAUnet/checkpoints/CP89_val_0.819235.pth'

    # net.cuda()
    net.load_state_dict(torch.load(model))
    #print(net)

    img_data = torch.from_numpy(img_data).type(torch.FloatTensor)
    a1, a2, a3, result = UNet_Mid(net, img_data)
    a1 = a1.cpu().detach().numpy()
    a2 = a2.cpu().detach().numpy()
    a3 = a3.cpu().detach().numpy()
    result = result.cpu().detach().numpy()
    #
    # result = result.detach().numpy()

    a1 = np.squeeze(a1)
    a1 = np.swapaxes(a1, 0, 2)
    #a1 = (a1 >= 0.5) + 0.
    a2 = np.squeeze(a2)
    a2 = np.swapaxes(a2, 0, 2)
    #a2 = (a2 >= 0.5) + 0.
    a3 = np.squeeze(a3)
    a3 = np.swapaxes(a3, 0, 2)
    #a3 = (a3 >= 0.5) + 0.
    result = np.squeeze(result)
    result = np.swapaxes(result, 0, 2)
    #result = (result >= 0.5)+0.

    print(result.shape)
    print(np.sum(result))

    a1nii = nib.Nifti1Image(a1, img_org_affine)
    a2nii = nib.Nifti1Image(a2, img_org_affine)
    a3nii = nib.Nifti1Image(a3, img_org_affine)
    resultnii = nib.Nifti1Image(result, img_org_affine)
    nib.save(a1nii, '004_right_000_a1_pre.nii.gz')
    nib.save(a2nii, '004_right_000_a2_pre.nii.gz')
    nib.save(a3nii, '004_right_000_a3_pre.nii.gz')
    nib.save(resultnii, '004_right_000_boundary_pre.nii.gz')