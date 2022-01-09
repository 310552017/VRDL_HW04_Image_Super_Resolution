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
4. [Inference](#Inference)

## Environment
requirement.txt contains all packages version of Python

## Dataset
- The training dataset contains 291 high resolution images.
- I transfer the training data to .h5 file to train the model.
- Run generate_train.m and you will get a file which is named as train.h5.

## Training
- Download the dataset train.h5 according to your path and run the file main_vdsr.py.
- To train the model, run the code as follow.
- The model weight will be saved in `checkpoint/` folder.
```bash
python main_vdsr.py 
```
```bash
usage: main_vdsr.py [-h] [--batchSize BATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--step STEP] [--cuda] [--resume RESUME]
               [--start-epoch START_EPOCH] [--clip CLIP] [--threads THREADS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--pretrained PRETRAINED] [--gpus GPUS]
               
optional arguments:
  -h, --help            Show this help message and exit
  --batchSize           Training batch size
  --nEpochs             Number of epochs to train for
  --lr                  Learning rate. Default=0.01
  --step                Learning rate decay, Default: n=10 epochs
  --cuda                Use cuda
  --resume              Path to checkpoint
  --clip                Clipping Gradients. Default=0.4
  --threads             Number of threads for data loader to use Default=1
  --momentum            Momentum, Default: 0.9
  --weight-decay        Weight decay, Default: 1e-4
  --pretrained PRETRAINED
                        path to pretrained model (default: none)
  --gpus GPUS           gpu ids (default: 0)
```
An example of training usage is shown as follows:
```
python main_vdsr.py --cuda --gpus 0
```

The training parameters are:

Model | learning rate | Training epochs | Batch size
------------------------ | ------------------------- | ------------------------- | -------------------------
CNN model | 0.001 | 25 | 750

### Link of my trained model
- The model which training with 100000 iterations：https://drive.google.com/file/d/14FpmiZiJ1SdBGayvGkJAg-5zSRXWL0jq/view?usp=sharing
- The model's training json file :https://drive.google.com/file/d/1Gp5-SdGiGUjhDb22cIQgHC0VIneMSgXY/view?usp=sharing

### Inference
Load the trained model parameters without retraining again.

“model_final.pth” need to be download to your own device and run “inference.ipynb” you will get the results as json file.
“model_final.pth” need to be put in the folder ./output/ that contains “inference.ipynb”.
