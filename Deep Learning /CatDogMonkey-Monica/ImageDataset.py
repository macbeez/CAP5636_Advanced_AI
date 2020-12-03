# Written by Lotzi Boloni March 15, 2020 --- changed July 27, 2020
# Loading and processing an image classification dataset, such as the DogCatMonkey 
# The dataset should be in a separate directory, under the subdirectory "data"
import os
import pathlib
import functools
# imports for visualization
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import PIL 

class DatasetCfg:
    """ A class collecting the configuration parameters for the dataset and the way it is presented for training"""
    
    def __init__(self, path_to_data):
        """ The path to data"""
        self.BATCH_SIZE = 32 #  was 32
        self.IMG_HEIGHT = 224
        self.IMG_WIDTH = 224
        # STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
        # training data is in a dir called training_data, each subdir is a class
        self.data_dir = pathlib.WindowsPath(path_to_data, "data")
        self.training_dir = pathlib.Path(self.data_dir, "training")
        self.training_count = len(list(self.training_dir.glob('*/*.jpg')))
        self.validation_dir = pathlib.Path(self.data_dir, "validation")
        self.validation_count = len(list(self.validation_dir.glob('*/*.jpg')))
        self.CLASS_NAMES = np.array([item.name for item in self.training_dir.glob('*') if item.name != "LICENSE.txt"])
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
    
def show_batch(cfg, image_batch, label_batch):
    """ visualize a batch of images from a dataset """
    plt.figure(figsize=(10,10))
    for n in range(min(25, len(image_batch))):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(cfg.CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
        
def get_label(cfg, file_path):
    """ Taking a file path, this will return a one-hot encoding of the label """
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == cfg.CLASS_NAMES

def decode_img(cfg, img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [cfg.IMG_WIDTH, cfg.IMG_HEIGHT])

def process_path(cfg, file_path):
    label = get_label(cfg, file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(cfg, img)
    return img, label

def prepare_for_training(cfg, ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    # Repeat forever
    ds = ds.repeat()
    ds = ds.batch(cfg.BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=cfg.AUTOTUNE)
    return ds


def create_dataset(cfg):
    list_ds = tf.data.Dataset.list_files(str(cfg.training_dir/'*/*'))

    ### test, can be removed 
    #for f in list_ds.take(5):
    #    print(f.numpy())
    #    print(get_label(cfg, f))
    
 
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    cfg_process_path = functools.partial(process_path, cfg)
    labeled_ds = list_ds.map(cfg_process_path, num_parallel_calls=cfg.AUTOTUNE)
    #labeled_ds = list_ds.map(process_path, num_parallel_calls=cfg.AUTOTUNE)
    
    ### test, can be removed
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())
    
    train_ds = prepare_for_training(cfg, labeled_ds)

    ### test, can be removed 
    #image_batch, label_batch = next(iter(train_ds))
    #show_batch(cfg, image_batch.numpy(), label_batch.numpy())

    # create the validation ds
    validation_list_ds = tf.data.Dataset.list_files(str(cfg.validation_dir/'*/*'))
    validation_labeled_ds = list_ds.map(cfg_process_path, num_parallel_calls=cfg.AUTOTUNE)
    validation_ds = prepare_for_training(cfg, validation_labeled_ds)

    return train_ds, validation_ds