This project contains the .onnx file that was used for the Expo App created online, and also contains the model training python file, specified with model weights and dataset links. The model Pytorch file (.pth) file has been added and can be used with the correct model defintion (specified in the script). However, the dataset cannot be directly added to GitHub. 



## Performance (Verified)

**Data Leakage Checks**
- Zero filename overlaps
- Zero duplicate images (MD5 verified)
- Clean train/validation split

**Results (1000-image holdout set):**
- AUC-ROC: 0.9501 (95.01%)
- Accuracy: 88.4%
- Sensitivity: 95.2%
- Precision: 83.8%
- NPV: 94.4%
