import tensorflow as tf
from glob import glob
import os
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import random
from clr import CyclicLR


os.environ["CUDA_VISIBLE_DEVICES"]="0"
sm.set_framework('tf.keras')

def parse_image(mask_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    img_path = tf.strings.regex_replace(mask_path, "_mask", "")
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    # image = tf.image.resize(image, (256,256))
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    #mask_path = glob(os.path.splitext(img_path)[0]+"*_mask.png")[0]
    #print(mask_path)
    #mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    #mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    # mask = tf.image.resize(mask, (256,256))
    # In scene parsing, "not labeled" = 255
    # But it will mess up with our N_CLASS = 150
    # Since 255 means the 255th class
    # Which doesn't exist
    #mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    # Note that we have to convert the new value (0)
    # With the same dtype than the tensor itself

    return {'image': image, 'segmentation_mask': mask}

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    input_mask = tf.cast(input_mask, dtype=tf.float32)


    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    if tf.random.uniform(()) > 0.4:
        input_image = tf.image.flip_up_down(input_image)
        input_mask = tf.image.flip_up_down(input_mask)

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.rot90(input_image, k=1)
        input_mask = tf.image.rot90(input_mask, k=1)

    if tf.random.uniform(()) > 0.45:
        input_image = tf.image.random_saturation(input_image, 0.7, 1.3)

    if tf.random.uniform(()) > 0.45:
        input_image = tf.image.random_contrast(input_image, 0.8, 1.2)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    input_mask = tf.cast(input_mask, dtype=tf.float32)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def dice_coe(output, target, axis = None, smooth=1e-10):
    output = tf.dtypes.cast( tf.math.greater(output, 0.5), tf. float32 )
    target = tf.dtypes.cast( tf.math.greater(target, 0.5), tf. float32 )
    inse = tf.reduce_sum(output * target, axis=axis)
    l = tf.reduce_sum(output, axis=axis)
    r = tf.reduce_sum(target, axis=axis)

    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice, name='dice_coe')
    return dice


def train(weights_paths, fold_number, model_name="unet", batch_size=16, loss_name="bce"):
    BATCH_SIZE = batch_size

    # for reference about the BUFFER_SIZE in shuffle:
    # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    BUFFER_SIZE = 1000

    dataset = {"train": train_dataset, "val": val_dataset}

    # -- Train Dataset --#
    dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
    dataset['train'] = dataset['train'].repeat()
    dataset['train'] = dataset['train'].batch(BATCH_SIZE)
    dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

    #-- Validation Dataset --#
    dataset['val'] = dataset['val'].map(load_image_test)
    dataset['val'] = dataset['val'].repeat()
    dataset['val'] = dataset['val'].batch(BATCH_SIZE)
    dataset['val'] = dataset['val'].prefetch(buffer_size=AUTOTUNE)

    print(dataset['train'])
    print(dataset['val'])


    if model_name=="unet":
        model = sm.Unet('efficientnetb2', input_shape=(None, None, 3), classes=N_CLASSES, activation='sigmoid',encoder_weights='imagenet',weights=weights_paths)
    if model_name=="fpn":
        model = sm.FPN('efficientnetb2', input_shape=(None, None, 3), classes=N_CLASSES, activation='sigmoid',encoder_weights='imagenet',weights=weights_paths)
    if model_name=="psp":
        model = sm.PSPNet('efficientnetb4', input_shape=(None, None, 3), classes=N_CLASSES, activation='sigmoid',encoder_weights=None)
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # 0.001

    if loss_name=="bce":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif loss_name=="bce_jaccard":
        loss = sm.losses.bce_jaccard_loss
    elif loss_name=="bce_jaccard_focal":
        loss = sm.losses.binary_focal_jaccard_loss
    elif loss_name=="binary_focal_dice":
        loss = sm.losses.binary_focal_dice_loss
     
    model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy',sm.metrics.iou_score,dice_coe])

    EPOCHS = 50

    STEPS_PER_EPOCH = TRAINSET_SIZE // BATCH_SIZE
    VALIDATION_STEPS = VALSET_SIZE // BATCH_SIZE


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint('results/weights/Fold'+str(fold_number)+'_'+str(model_name)+'_'+str(loss_name)+'.h5', monitor='val_dice_coe', mode='max', verbose=1, save_best_only=True, save_weights_only=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_dice_coe', factor=0.1, patience=8, min_lr=0.00001,mode='max')
    ]
    
    results = model.fit(dataset['train'], epochs=EPOCHS,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            validation_steps=VALIDATION_STEPS,
                            callbacks=callbacks,
                            validation_data=dataset['val'])

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend();
    plt.savefig('./results/plots/train_loss_Fold'+str(fold_number)+'_'+str(model_name)+'_'+str(loss_name)+'.png')

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["dice_coe"], label="dice_coe")
    plt.plot(results.history["val_dice_coe"], label="val_dice_coe")
    plt.plot( np.argmax(results.history["val_dice_coe"]), np.max(results.history["val_dice_coe"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coeff")
    plt.legend();
    plt.savefig('./Results/plots/train_dice_Fold'+str(fold_number)+'_'+str(model_name)+'_'+str(loss_name)+'.png')

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["iou_score"], label="iou_score")
    plt.plot(results.history["val_iou_score"], label="val_iou_score")
    plt.plot( np.argmax(results.history["val_iou_score"]), np.max(results.history["val_iou_score"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("IOU")
    plt.legend();
    plt.savefig('./Results/plots/train_IOU_Fold'+str(fold_number)+'_'+str(model_name)+'_'+str(loss_name)+'.png')

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["accuracy"], label="accuracy")
    plt.plot(results.history["val_accuracy"], label="val_accuracy")
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.legend();
    plt.savefig('./Results/plots/train_accuracy_Fold'+str(fold_number)+'_'+str(model_name)+'_'+str(loss_name)+'.png')


SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE


dataset_path = "dataset/patches_all/"


IMG_SIZE = 256
N_CHANNELS = 3
N_CLASSES = 1


files = glob(dataset_path + "*/*_mask.png")
random.shuffle(files)
files = np.array(files)
BATCH_SIZE = 16

from sklearn.model_selection import KFold


kf = KFold(n_splits=5)
for idx, (train_idx, test_idx) in enumerate(kf.split(files)):
    print("Training Fold ",str(idx+1))
    print(f"The Training Dataset contains {train_idx.shape[0]} images.")
    print(f"The Validation Dataset contains {test_idx.shape[0]} images.")
    TRAINSET_SIZE = train_idx.shape[0]
    VALSET_SIZE = test_idx.shape[0]

    train_files, test_files = files[train_idx], files[test_idx]
    train_files = train_files.tolist()
    test_files = test_files.tolist()

    train_list_ds = tf.data.Dataset.from_tensor_slices(train_files)
    test_list_ds = tf.data.Dataset.from_tensor_slices(test_files)

    train_dataset = train_list_ds.map(parse_image)
    val_dataset = test_list_ds.map(parse_image)

    # train(model_name="fpn",batch_size=BATCH_SIZE,loss_name="bce",weights_paths=None, fold_number=idx+1) # "results/weights/unet_bce_pretrain.h5"
    train(model_name="unet",batch_size=BATCH_SIZE,loss_name="bce",weights_paths=None, fold_number=idx+1) # "results/weights/unet_bce_pretrain.h5"
    



