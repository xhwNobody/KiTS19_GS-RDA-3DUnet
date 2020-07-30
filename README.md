# KiTS19_GS-RDA-Unet
This is a part of soultion to deal with the KiTS19 Challenge which aim is to segment kidney and tumor.You can get the detail by https://kits19.grand-challenge.org/.
## Introduction
The solution is based on nn-Unet and also make some changes.This repository only includes the use of CNN for training and inference on pre-processed voxels, and does not include pre-processing and post-processing.
## Data Augmentations
Using the batchgenerators to get more samples makes the model more generalized.
## Corse Kidney Segmentation
The first step is to roughly segment the kidney and tumor areas.The adopted model introduces the residual dual attention module on the basis of Unet, so it is called RDA-Unet，which is described in first_step_kidney_segmentation/build_RDA_3DUnet.py.
