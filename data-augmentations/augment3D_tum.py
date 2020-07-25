import numpy as np
import nibabel as nib

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import  Compose
from batchgenerators.transforms.spatial_transforms import  SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform

def get_list_of_patients(data_floder):
    nii_files = subfiles(data_floder, suffix='.nii.gz', join=True)
    patients = [i[-24:-11] for i in nii_files]
    return patients

class Kits2019DataLoader3D(DataLoader):
    def __init__(self, data, batch_size, patch_size, num_threads_in_multithreaded, seed_for_shuffle=1234,
                 return_incomplete=False, shuffle=False, infinite=True):
        super().__init__(data, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle, infinite)
        self.patch_size = patch_size
        self.num_modalities = 1
        self.indices = list(range(len(data)))

    @staticmethod
    def load_patient(patient):
        data = nib.load('/home/xhw/桌面/data-augmentations/val_image_data/' + patient +'_img.nii.gz')
        data = data.get_data()
        metadata = nib.load('/home/xhw/桌面/data-augmentations/val_mask_data/' + patient+'_msk.nii.gz')
        metadata = metadata.get_data()
        return data, metadata

    def generate_train_batch(self):
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]
        #print(patients_for_batch)
        #print('***')

        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patients_names = []
        for i,j in enumerate(patients_for_batch):
            patient_data, patient_metadata = self.load_patient(j)
            patient_data = np.expand_dims(patient_data,0)
            patient_seg = np.expand_dims(patient_metadata, 0)

            data[i] = patient_data
            seg[i] = patient_seg

            metadata.append(patient_metadata)
            patients_names.append(j)
        return {'data':data, 'seg':seg, 'metadata':metadata, 'names':patients_names}

def get_train_transform(patch_size):
    tr_transforms = []
    
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i//2 for i in patch_size],
            do_elastic_deform=True, deformation_scale=(0, 0.05),
            do_rotation=True,
            angle_x=(-5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
            angle_y=(-5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
            angle_z = (-5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.75,1.25),
            border_mode_data='constant', border_cval_data=-2.34,
            border_mode_seg='constant', border_cval_seg=0
        )
    )
    
    tr_transforms.append(MirrorTransform(axes=(0,1,2)))
    tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))
    tr_transforms.append(GaussianNoiseTransform(noise_variance=(0,0.15), p_per_sample=0.15))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms

data_path = '/home/xhw/桌面/data-augmentations/val_image_data'
dst_img = '/home/xhw/桌面/data-augmentations/val_image_data_aug/'
dst_msk = '/home/xhw/桌面/data-augmentations/val_mask_data_aug/'
patients = get_list_of_patients(data_path)
batch_size = 2
patch_size = (128, 128, 48)
kitsloader = Kits2019DataLoader3D(patients, batch_size, patch_size, 1)
tr_transforms = get_train_transform(patch_size)
tr_gen = MultiThreadedAugmenter(kitsloader, tr_transforms, seeds=None, pin_memory=False, num_processes=1)


demo_affine = nib.load('/home/xhw/桌面/data-augmentations/val_image_data/000_left_000_img.nii.gz').affine
for i in range(2209):
    for j in range(2):
        tr_gen_batch = next(tr_gen)
        dat, seg = tr_gen_batch['data'][j], tr_gen_batch['seg'][j]
        dat = np.squeeze(dat)
        seg = np.squeeze(seg)
        img2 = nib.Nifti1Image(dat, demo_affine)
        seg2 = nib.Nifti1Image(seg, demo_affine)
        nib.save(img2, dst_img + 'aug_'+str(i)+'_'+str(j)+'_img.nii.gz')
        nib.save(seg2, dst_msk + 'aug_'+str(i)+'_'+str(j)+'_msk.nii.gz')














