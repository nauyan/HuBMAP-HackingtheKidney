import tifffile as tiff
import numpy as np
import pandas as pd
import os
from skimage.util.shape import view_as_windows
import cv2
import shutil
from tqdm import tqdm

def init_directories(directory_path):
    
    if os.path.exists(directory_path):
        try:
            os.rmdir(directory_path)
        except:
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
    else:
        os.makedirs(directory_path)
    

def enc2mask(encs, shape):
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for m,enc in enumerate(encs):
        if isinstance(enc,np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1 + m
    return img.reshape(shape).T

def mask2enc(mask, shape, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1,n+1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0: encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs
    
def save_sample(idx, sample, dir_path):
    for idx1, sample in enumerate(sample):
        p_th = 1000*(new_size//256)**2
        s_th = 40

        img = cv2.resize(sample[:,:,:3],(new_size,new_size),interpolation = cv2.INTER_AREA)
        mask = cv2.resize(sample[:,:,3],(new_size,new_size),interpolation = cv2.INTER_AREA)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        if (s>s_th).sum() <= p_th or img.sum() <= p_th:
            continue 

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(dir_path + str(idx) + "_" + str(idx1) + ".png", img)
        cv2.imwrite(dir_path + str(idx) + "_" + str(idx1) + "_mask.png", mask)

directory_train = "dataset/patches_all/labels/"

init_directories(directory_train)

MASKS = 'dataset/train.csv'
DATA = 'dataset/train/'
patch_size = 1024
new_size = 256
df_masks = pd.read_csv(MASKS).set_index('id')


print("Generating Patches")
for index, encs in tqdm(df_masks.iterrows(),total=len(df_masks)):
    img = tiff.imread(os.path.join(DATA,index+'.tiff'))
    if len(img.shape) == 5:img = np.transpose(img.squeeze(), (1,2,0))
    if img.shape[0]==3:img = np.transpose(img.squeeze(), (1,2,0))
    mask = enc2mask(encs,(img.shape[1],img.shape[0]))
    mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

    sample = view_as_windows(np.concatenate((img, mask), axis=2),(patch_size,patch_size,4),(patch_size,patch_size,4)).reshape(-1,patch_size,patch_size,4)
    save_sample(index,sample,directory_train)
