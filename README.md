# VRDL_HW04_Image_Super_Resolution
Dataset:
Training set: 291 high-resolution images
Testing set: 14 low-resolution images
Train your model to reconstruct a high-resolution image from a low-resolution input
Pre-trained model is NOT allowed!


## Coding Environment
- Python

## Reproducing Submission
To reproduct the testing prediction, please follow the steps below:
1. [Environment](#environment)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Testing](#testing)

## Environment
requirement.txt contains all packages version of Python

## Dataset
- The training dataset contains 291 high resolution images.
- I transfer the training data to .h5 file to train the model.
- Run generate_train.m and you will get a file which is named as train.h5.

## Training
- Download the dataset train.h5 according to your path and run the file main_vdsr.py.
- Run the files "VRDL_HW03.ipynb" will start to train the model and save it as "model_final.pth".
- Remember to replace the root of the image file with your own root.

The training parameters are:

Model | learning rate | Training iterations | Batch size
------------------------ | ------------------------- | ------------------------- | -------------------------
MaskRCNN_resnet101_fpn | 0.00025 | 100000 | 2

## Testing
- "VRDL_HW03.ipynb" has the code that can use the model which is saved above to predict the testing images and save the prediction result as json files according to coco set rules.

### Pretrained models
Pretrained model "MaskRCNN_resnet101_fpn" which is provided by detectron2.

### Link of my trained model
- The model which training with 100000 iterations：https://drive.google.com/file/d/14FpmiZiJ1SdBGayvGkJAg-5zSRXWL0jq/view?usp=sharing
- The model's training json file :https://drive.google.com/file/d/1Gp5-SdGiGUjhDb22cIQgHC0VIneMSgXY/view?usp=sharing

### Inference

Load the trained model parameters without retraining again.

“model_final.pth” need to be download to your own device and run “inference.ipynb” you will get the results as json file.
“model_final.pth” need to be put in the folder ./output/ that contains “inference.ipynb”.
