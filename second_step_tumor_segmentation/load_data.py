import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data.dataset import Dataset

class Kits19Dataset(Dataset):
    def __init__(self, imgPath, mskPath, edgPath):
        self.imgNameList = os.listdir(imgPath)
        self.imgPath = imgPath
        self.mskPath = mskPath
        self.edgPath = edgPath

    def __getitem__(self, index):
        img = nib.load(self.imgPath + self.imgNameList[index]).get_data()
        img = np.expand_dims(img, -1)
        mask = nib.load(self.mskPath + self.imgNameList[index][:-10] + 'msk.nii.gz').get_data()
        edge = nib.load(self.edgPath + self.imgNameList[index][:-10] + 'edg.nii.gz').get_data()

        mask1 = np.expand_dims(mask, -1)
        mask2 = np.expand_dims(mask, -1)
        mask3 = np.expand_dims(mask, -1)
        mask_1 = (mask1 == 0.) + 0
        mask_2 = (mask2 == 1.) + 0
        mask_3 = (mask3 == 2.) + 0
        mask = np.concatenate((mask_1, mask_2, mask_3), -1)
        mask = np.array(mask)
        edge = np.expand_dims(edge, -1)
        edge = (edge == 2.) + 0

        img = np.swapaxes(np.swapaxes(img, 0, 3), 1, 2)
        mask = np.swapaxes(np.swapaxes(mask, 0, 3), 1, 2)
        gt_edge = np.swapaxes(np.swapaxes(edge, 0, 3), 1, 2)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        gt_edge = torch.from_numpy(gt_edge).type(torch.FloatTensor)

        return img, mask, gt_edge

    def __len__(self):
        return len(self.imgNameList)








