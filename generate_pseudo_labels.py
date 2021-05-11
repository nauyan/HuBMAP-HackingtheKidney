import glob
import tifffile as tiff
import numpy as np
from skimage.util.shape import view_as_windows
import os
import cv2
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="0"
import segmentation_models as sm
from tqdm import tqdm
import tensorflow as tf

sm.set_framework('tf.keras')

def init_directories(directory_path):
    
    if os.path.exists(directory_path):
        try:
            os.rmdir(directory_path)
        except:
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
    else:
        os.makedirs(directory_path)

def load_models():
    fold_models = []
    for fold_model_path in glob.glob("results/weights/*unet*.h5"):
        if "_new" in fold_model_path:
            continue
        print(fold_model_path)
        fold_models.append(tf.keras.models.load_model(fold_model_path,compile=False))
    return fold_models

def save_sample(idx, sample, dir_path, fold_models):
    for idx1, sample in enumerate(sample):
        p_th = 1000*(new_size//256)**2
        s_th = 40

        img = cv2.resize(sample,(new_size,new_size),interpolation = cv2.INTER_AREA)


        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)


        if (s>s_th).sum() <= p_th or img.sum() <= p_th:
            # print((s>s_th).sum(),img.sum())
            continue 

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        
        pred = None

        for fold_model in fold_models:
            if pred is None:
                pred = np.squeeze(fold_model.predict(np.expand_dims(img/255.0, 0)))
            else:
                pred += np.squeeze(fold_model.predict(np.expand_dims(img/255.0, 0)))

        pred = pred/len(fold_models)

        pred[pred>0.30] = 1
        pred[pred<0.30] = 0

        count = np.count_nonzero(pred==0)+np.count_nonzero(pred==1)
        total_pixels = 256*256
        
        
        if count/total_pixels<0.95:
            continue
        
        cv2.imwrite(dir_path + str(idx) + "_" + str(idx1) + ".png", img)
        cv2.imwrite(dir_path + str(idx) + "_" + str(idx1) + "_mask.png", pred)

test_wsis_path = glob.glob("dataset/test/*.tiff")
directory_train = "dataset/patches_all/pseudo_labels/"

init_directories(directory_train)
fold_models = load_models()

patch_size = 1024
new_size = 256

for wsi_path in tqdm(test_wsis_path,total=len(test_wsis_path)):
    index = os.path.basename(wsi_path).split('.')[0]

    img = tiff.imread(wsi_path)
    if len(img.shape) == 5:img = np.transpose(img.squeeze(), (1,2,0))
    if img.shape[0]==3:img = np.transpose(img.squeeze(), (1,2,0))


    sample = view_as_windows(img,(patch_size,patch_size,3),(patch_size,patch_size,3)).reshape(-1,patch_size,patch_size,3)
    save_sample(index,sample,directory_train,fold_models)