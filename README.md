# KiTS19_GS-RDA-3DUnet
This repository provides a part of soultion to deal with the KiTS19 Challenge which aim is to segment kidney and tumor.You can get the detail by https://kits19.grand-challenge.org/. You can use it to train quickly and easily
## Introduction
The solution is based on nn-Unet and also make some changes.This repository only includes the use of CNN for training and inference on pre-processed voxels, and does not include pre-processing and post-processing.
## Data Augmentations
Using the batchgenerators to get more samples makes the model more generalized.
## Corse Kidney-Tumor Segmentation
The first step is to roughly segment the kidney and tumor areas.The adopted model introduces the residual dual attention module on the basis of Unet, so it is called RDA-Unet，which is described in first_step_kidney_segmentation/build_RDA_3DUnet.py.Training data can be obtained from here.
## Fine Tumor Segmentation
The second step is to further segment the tumor area on the basis of the first step. The adopted model introduces gated shape convolution based on the RDA-Unet model, and designs a gated shape sub-network to predict the boundary of the tumor, thereby improving the accuracy of tumor segmentation.Training data can be obtained from here.
## Result

## Reference
https://github.com/nv-tlabs/GSCNN
