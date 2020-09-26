# -*- coding: utf-8 -*-
from os import path

# define the paths to the training and validation directories
TRAIN_IMAGES = "../datasets/tiny-imagenet-200/train"
VAL_IMAGES = "../datasets/tiny-imagenet-200/val/images"

# define the
# their corresponding class labels
VAL_MAPPINGS = "../datasets/tiny-imagenet-200/val/val_annotations.txt"

# define the paths to the WordNet hierarchy files which are used
# to generate our class labels
WORDNET_IDS = '../datasets/tiny-imagenet-200/wnids.txt'
WORD_LABELS = '../datasets/tiny-imagenet-200/words.txt'

# since we do not have access to the testing data we need to
# take a number of images from the training data and use it instead
#NUM_CLASSES = 200 
NUM_CLASSES = 2 #tuan cheat here
NUM_TEST_IMAGES = 50 * NUM_CLASSES

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "../datasets/tiny-imagenet-200/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/tiny-imagenet-200/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/tiny-imagenet-200/hdf5/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "output/tiny-image-net-200-mean.json"

# define the path to the output directory used for storing plots,
# classification reports, etc.
OUTPUT_PATH = "output"
MODEL_PATH = path.sep.join([OUTPUT_PATH,
                            "checkpoints/epoch_20.hdf5"])
FIG_PATH = path.sep.join([OUTPUT_PATH,
                          'deepergooglenet_tinyimagenet.png'])
JSON_PATH = path.sep.join([OUTPUT_PATH,
                           'deepergooglenet_tinyimagenet.json'])

EXPERIMENT_CONFIGS = {
  "e11": [25,'SGD',1e-2,"python train.py --checkpoints output/checkpoints --experiment 11"],
 "e12": [10,'SGD',1e-3,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_25.hdf5 --start_epoch 25 --experiment 12"],
 "e13": [30,'SGD',1e-4,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_35.hdf5 --start_epoch 35 --experiment 13"],
 "e21": [20,'Adam',1e-3,"python train.py --checkpoints output/checkpoints --experiment 21"],
 "e22": [10,'Adam',1e-4,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_20.hdf5 --start_epoch 20 --experiment 22"],
"e23": [10,'Adam',1e-5,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_30.hdf5 --start_epoch 30 --experiment 23"],
 "e31": [40,'Adam',1e-3,"python train.py --checkpoints output/checkpoints --experiment 31"],
 "e32": [20,'Adam',1e-4,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_40.hdf5 --start_epoch 40 --experiment 32"],
"e33": [10,'Adam',1e-5,"python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_60.hdf5 --start_epoch 60 --experiment 33"],  
}