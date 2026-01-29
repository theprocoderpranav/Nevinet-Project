This project contains the .onnx file that was used for the Expo App created online, and also contains the model training python file, specified with model weights and dataset links. The model Pytorch file (.pth) file has been added and can be used with the correct model defintion (specified in the script). However, the dataset cannot be directly added to GitHub. 
Datasets that were used:
1. PAD_UFES_20 Skin Imaging Dataset consiting of smartphone-captured lesions. 
2. HAM10000 (Human Against Machine with ~10,000 images of skin lesions and clear dermoscopic imaging)
Total dataset size: 
2,400 images for training
800 images for validation and testing.


## Performance (Verified)

**Data Leakage Checks**
- Zero filename overlaps
- Zero duplicate images (MD5 verified)
- Clean train/validation split

**Results (400-image holdout set):**
- AUC-ROC: 0.9511 (95.11%)
- Accuracy: 88.25%
- Sensitivity: 90.0%
- Specificity: 86.5%
- NPV: 89.6%

**Validation Set:**
- 62% dermoscopy images (HAM10000)
- 32% clinical photos (PAD-UFES-20)
- 4% unkown/unclear (Probably clinical photos)
- Mixed difficulty, real-world conditions

**Tested on:**
- Modern smartphones (Pixel, Galaxy)
- Legacy hardware (2010 phone)
- Various image qualities and resolutions
