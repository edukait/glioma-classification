# Brain Tumor Segmentation and Classification
The purpose of this project is to be able to automatically and efficiently segment and classify high-grade and low-grade gliomas. MRI images from the BraTS 2018 contest is used for this project, as the data is already preprocessed and skull-stripped.
## Segmentation
The segmentation code implements a convolutional neural network with a U-Net architecture. Part of the segmentation code requires extracting patches from the MRI images in order to speed up the computation.
## Classification
The second part of the project uses both a support vector machine and deep neural network in order to compare the efficiency of both models relative to each other. The features of the tumors segmented with the segmentation code will be extracted using Discrete Wavelet Transform, and then will undergo PCA before being fed into the classification models.
