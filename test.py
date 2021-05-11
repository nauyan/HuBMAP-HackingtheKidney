import glob
import cv2

import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from tensorflow import nn
import tensorflow as tf
from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import shape
import numpy as np
import tensorflow_addons as tfa

import segmentation_models as sm

from sklearn.metrics import confusion_matrix,classification_report
import progressbar


model = tf.keras.models.load_model('results/weights/fpn.h5',compile=False)

test_samples = glob.glob("dataset/patches/val/*_mask.png")
predictions = []
masks = []
count = 0
for sample in progressbar.progressbar(test_samples):
    img = cv2.imread(sample.replace("_mask",""),-1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    
    


    mask = cv2.imread(sample,-1)

    pred = model.predict(img)
    pred = (pred > 0.5).astype(np.uint8)
    pred = np.squeeze(pred)
    pred = cv2.resize(pred, (1024,1024), interpolation = cv2.INTER_AREA)

    predictions.append(pred)
    masks.append(mask)
    count = count + 1


print(len(predictions),len(masks))
predictions = np.array(predictions).ravel()
masks = np.array(masks).ravel()
print(predictions.shape,masks.shape)

target_names = ['class 0', 'class 1']
print(classification_report(masks,predictions,labels=[0,1]))
print(confusion_matrix(masks,predictions))