# HuBMAP-HackingtheKidney

Competition Homepage: https://www.kaggle.com/c/hubmap-kidney-segmentation/data

## Dataset
The dataset is comprised of very large (>500MB - 5GB) TIFF files. The training set has 8, and the public test set has 5. The private test set is larger than the public test set.

The training set includes annotations in both RLE-encoded and unencoded (JSON) forms. The annotations denote segmentations of glomeruli.

Both the training and public test sets also include anatomical structure segmentation. They are intended to help you identify the various parts of the tissue.

## Installation
To run the code seemlessly you can install all the repo assoacited packages usign [HuBMAP.yml](HuBMAP.yml) file. 

To install all the packages use the following command:
```
conda env create -f environment.yml
```

## Patch Creation
Since the test and train images are very large to fit in memory we create small patches from large images. We generate overlapping images of size 1024x1024 and later resize them to 256x256 pixels along with the associated ground truth masks. 

Patch Generation Code can be ran using:
```
python create_patches.py
```
## Pseudo Labelling
The number of Glomeruli in the training set is very low therefore to improve model generalization we use Pseudo Labelling to increase the model performance. The inspiration of Pseudo Labelling was inspired by a talk of [Yauhen Babakhin](https://www.youtube.com/watch?v=SsnWM1xWDu4) in which multiple iterations of Model Predictions are carried out on Public Test set are done to refine synthetically generated pseudo labels. We used an ensemble of U-Net(Efficient-B2 and Efficient-B4) for the generation of pseudo labels.
```
python generate_pseudo_labels.py
```
## Training
We have used U-Net and FPN with various backbones for training. The parameters we have used for submission in the HuBMAP competition can be seen in the [train.py](train.py) and [train_Fold.py](train_Fold.py) files. The [train.py](train.py) comprises the model trained for a single fold of data and [train_Fold.py](train_Fold.py) comprises the model trained for 5 Fold Cross-validation. To update the training and model parameters please refer to the training files.

To start 1 Fold training use: 
```
python train.py
```
To start 5 Fold training use:
```
python train_Fold.py
```

## Testing
The test script is used to calculate the evaluation metrics for model perfpoamnce on validation data.
To start test script use:
```
python test.py
```

## Quantitative Results

| Model | Dice Score | Dice Score |
| ----- | ---- | ---- |
| Unet | 0.7906 | Dice Score 
| Segnet | 0.3684 | Dice Score 
| DeeplabV3+ | 0.7743 | Dice Score 
| Unet + Skip Connections + ASPP + SE Block| 0.8005 | Dice Score 
