import numpy as np
from skimage import data
import matplotlib.pyplot as plt

from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoaderBase

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.abstract_transforms import  Compose
from batchgenerators.transforms.spatial_transforms import  SpatialTransform_2
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform


class DataLoader(DataLoaderBase):
    def __init__(self, data, BATCH_SIZE=2, num_batches=None, seed=False):
        super(DataLoader, self).__init__(data, BATCH_SIZE, num_batches, seed)

    def generate_train_batch(self):
        img = self._data
        img = np.tile(img[None, None], (self.BATCH_SIZE,1 ,1, 1))
        return {'data':img.astype(np.float32), 'sole_other_key':'some other value'}

batchgen = DataLoader(data.camera(), 1, None, False)
#batch = next(batchgen)

#print(batch['data'].shape)
def plot_batch(batch):
    batch_size = batch['data'].shape[0]
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        plt.imshow(batch['data'][i, 0], cmap="gray")
    plt.show()
#plot_batch(batch)

my_transforms = []

brightness_transform = ContrastAugmentationTransform((0.3, 3.), preserve_range=True)
my_transforms.append(brightness_transform)

noise_transform = GaussianNoiseTransform(noise_variance=(0, 20)) ##
my_transforms.append(noise_transform)

spatial_transform = SpatialTransform_2(data.camera().shape, np.array(data.camera().shape)//2,
                                     do_elastic_deform=True, deformation_scale=(0,0.05),
                                     do_rotation=True, angle_z=(0, 2*np.pi),
                                     do_scale=True, scale=(0.8, 1.2),
                                     border_mode_data='constant', border_cval_data=0, order_data=1,
                                     random_crop=False)
my_transforms.append(spatial_transform)
all_transforms = Compose(my_transforms)
multithreaded_generator = MultiThreadedAugmenter(batchgen, all_transforms, 4, 2, seeds=None)
plot_batch(next(multithreaded_generator))
