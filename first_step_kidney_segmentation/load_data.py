import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data.dataset import Dataset

class Kits19Dataset(Dataset):
    def __init__(self, imgPath, mskPath):
        self.imgNameList = os.listdir(imgPath)
        self.imgPath = imgPath
        self.mskPath = mskPath

    def __getitem__(self, index):
        img = nib.load(self.imgPath + self.imgNameList[index]).get_data()
        img = np.expand_dims(img, -1)
        mask = nib.load(self.mskPath + self.imgNameList[index][:-10] + 'msk.nii.gz').get_data()

        mask1 = np.expand_dims(mask, -1)
        mask2 = np.expand_dims(mask, -1)
        mask3 = np.expand_dims(mask, -1)
        mask_1 = (mask1 == 0.) + 0
        mask_2 = (mask2 == 1.) + 0
        mask_3 = (mask3 == 2.) + 0
        mask = np.concatenate((mask_1, mask_2, mask_3), -1)
        mask = np.array(mask)

        img = np.swapaxes(np.swapaxes(img, 0, 3), 1, 2)
        mask = np.swapaxes(np.swapaxes(mask, 0, 3), 1, 2)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        return img, mask

    def __len__(self):
        return len(self.imgNameList)








