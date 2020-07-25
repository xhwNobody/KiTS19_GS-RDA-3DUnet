import os
import numpy as np
import nibabel as nib
from build_GS_RDAunet import *

test_img_path = '/home/xhw/桌面/kits19-tumor-GS-RDAUnet/atest/test_data/'
dst_pred_path = '/home/xhw/桌面/kits19-tumor-GS-RDAUnet/atest/test_pred_GSRDAUnet/'
use_gpu = 1

net = UNet(n_channels=1, n_classes=3)
model = '/home/xhw/桌面/kits19-tumor-GS-RDAUnet/checkpoints/CP95_val_0.821611.pth'
net.cuda()
net.load_state_dict(torch.load(model))

for img_nii in os.listdir(test_img_path):
    print(img_nii)
    #if not os.path.exists(dst_pred_path + img_nii[:-15] + '_pre'):
       #os.mkdir(dst_pred_path + img_nii[:-15]+ '_pre')
    
    imgaffine = nib.load(test_img_path + img_nii).affine
    imgData = nib.load(test_img_path + img_nii).get_data()
    imgData = np.expand_dims(imgData, 0)
    imgData = np.expand_dims(imgData, -1)

    imgData = np.swapaxes(np.swapaxes(imgData, 1, 4), 2, 3)
    imgData = torch.from_numpy(imgData).type(torch.FloatTensor)

    net.eval()
    if use_gpu:
        imgData = imgData.cuda()
    #print(imgData.shape)
    #msk_pred = net(imgData).cpu().detach().numpy()
    msk_pred,shape_pred = net(imgData)
    msk_pred = msk_pred.cpu().detach().numpy()
    #print(msk_pred.shape)
    msk_pred = np.swapaxes(np.swapaxes(msk_pred, 1, 4), 2, 3)
    msk_pred = np.squeeze(msk_pred, 0)
    msk_pred = np.argmax(msk_pred,axis=3)
    mskdata = nib.Nifti1Image(msk_pred.astype(np.float), imgaffine)
    nib.save(mskdata, dst_pred_path + img_nii[:-10]+ 'pre.nii.gz')
    




