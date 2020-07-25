import os
import numpy as np
import nibabel as nib
from build_model import *

test_img_path = '/home/xhw/桌面/kits19-experient-pytorch/atest/test_data/imagenii/'
dst_pred_path = '/home/xhw/桌面/kits19-experient-pytorch/atest/test_pred/'
use_gpu = 1

net = UNet(n_channels=1, n_classes=2)
model = '/home/xhw/桌面/kits19-experient-pytorch/atest/CP87_val_0.979190.pth'
net.cuda()
net.load_state_dict(torch.load(model))

for img_nii in os.listdir(test_img_path):
    print(img_nii)
    if not os.path.exists(dst_pred_path + img_nii[:3]):
       os.mkdir(dst_pred_path + img_nii[:3])
    
    imgaffine = nib.load(test_img_path + img_nii).affine
    imgData = nib.load(test_img_path + img_nii).get_data()
    imgData = np.expand_dims(imgData, 0)
    imgData = np.expand_dims(imgData, -1)

    imgData = np.swapaxes(np.swapaxes(imgData, 1, 4), 2, 3)
    imgData = torch.from_numpy(imgData).type(torch.FloatTensor)

    net.eval()
    if use_gpu:
        imgData = imgData.cuda()
    msk_pred = net(imgData).cpu().detach().numpy()
    msk_pred = np.swapaxes(np.swapaxes(msk_pred, 1, 4), 2, 3)
    msk_pred = np.squeeze(msk_pred, 0)
    mskdata = nib.Nifti1Image(msk_pred, imgaffine)
    nib.save(mskdata, dst_pred_path + img_nii[:3] + '/' + img_nii[:-10]+ 'pre.nii.gz')
    




