import os
import numpy as np
import shutil
import nibabel as nib

org_img_path = '/media/xhw/Newsmy/datatrain&test/train_data/image/'
org_msk_path = '/media/xhw/Newsmy/datatrain&test/train_data/mask/'
dst_img_path = './with_tumor_img/'
dst_msk_path = './with_tumor_msk/'

mask_files = os.listdir(org_msk_path)

for mask_file in mask_files:
    mask = nib.load(org_msk_path+mask_file)
    mask_data = mask.get_data()
    maxnum = np.max(mask_data)
    if maxnum==2:
       shutil.copyfile(org_msk_path+mask_file, dst_msk_path+mask_file)
       shutil.copyfile(org_img_path+mask_file[:-11]+'_img.nii.gz', dst_img_path+mask_file[:-11]+'_img.nii.gz')
       
