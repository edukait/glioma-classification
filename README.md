# Brain Tumor Segmentation and Classification
The purpose of this project is to be able to automatically and efficiently segment and classify high-grade and low-grade gliomas. MRI images from the BraTS 2018 contest is used for this project, as the data is already preprocessed and skull-stripped.

## Segmentation
The segmentation code implements a convolutional neural network with a U-Net architecture. Part of the segmentation code requires extracting patches from the MRI images in order to speed up the computation. The following image is a schematic depicting the general process used for extracting patches from the MRI scans and feeding them into the convolutional neural network.

<p align="center">
  <img src="/misc/patch_extraction.jpg" width="300">
</p>

The Dice scores, which were the metric used for quantifying the accuracy of the CNN, ranged anywhere from 0.7 to 0.85.

## Classification
The second part of the project uses both a support vector machine and deep neural network in order to compare the efficiency of both models relative to each other. The features of the tumors segmented with the segmentation code will be extracted using the feature extractor from the open-sourced [PyRadiomics](https://github.com/Radiomics/pyradiomics) package. Based on past readings of papers, the following features were extracted for further analysis:

1. Elongation: Shows the relationship between the two largest principal components in the Region of Interest (ROI) shape.
2. Flatness: Shows the relationship between the largest and smallest principal components in the ROI shape.
3. Major Axis Length: The largest axis length of the ROI-enclosing ellipsoid.
4. Minor Axis Length: The second-largest axis length of the ROI-encloding ellipsoid.
5. Maximum 3D Diameter: The largest pairwise Euclidean distance between tumor surface mesh vertices (also known as Feret Diameter).
6. Sphericity: Measure of the roundness of the shape of the tumor region relative to a sphere.
7. Surface Area: Surface area of the mesh in millimeters squared.
8. Energy: Measure of the magnitude of voxel values in an image.
9. Entropy: Specifices the uncertainty/randomness in the image values.
10. Kurtosis: Measure of the "peakedness" of the distribution of values in the image ROI.
11. Mean: Mean of the intensity values of the pixels.
12. Skewness: Measures the asymmetry of the distribution of values about the Mean value.
13. Contrast: Measure of the local intensity variation.
14. Correlation: Value between 0 and 1 showing the linear dependency of gray level values to their respective voxels in the GLCM.
15. Complexity: Measure of the amount of primitive components in the image.
16. Strength: Measure of the primitives in an image.

The remainder of the code compares the performance of two models, which are support vector classifiers and random forest classifiers. In order to quantify the accuracy of the models, six main metrics were used, which were all calculated after model averaging and cross-validation:

1. Accuracy: The proportion of correct predictions.
2. Classification Error: The proportion of incorrect predictions.
3. Sensitivity: The proportion of correctly identified true positives.
4. Specificity: The proportion of correctly identified true negatives.
5. False Positive Rate: The proportion of negatives falsely identified as positives.
6. Precision: The proportion of true positives to the total number of elements identified as positives.

## Conclusion

| | Support Vector Classifier | Random Forest Classifier |
| - | :-------------------------: | :------------------------: |
| Accuracy Score | 0.81 | 0.83 |
| Classification Error | 0.19 | 0.17 |
| Sensitivity | 0.95 | 1.0 |
| Specificity | 0.63 | 0.60 |
| False Positive Rate | 0.38 | 0.40 |
| Precision | 0.79 | 0.78 |

The support vector classifier and random forest classifier exhibited similar performance, as the metric measurements yield similar results. Shown belown is a graphical representation of the table displayed above:

<p align="center">
  <img src="/misc/model_comparison.png" width="500">
</p>

## Future Directions
Some directions for the near future include improving the current classification models and implementing more models to see how they perform. In the distant future, however, I hope to develop a frontend and make this tool easily accessible and usable for the general public.
