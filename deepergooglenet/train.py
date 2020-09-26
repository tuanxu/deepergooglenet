# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import tiny_imagenet_config as config
from pyimagesearch.preprocessing import imagetoarraypreprocessor as ITA
from pyimagesearch.preprocessing import simplespreprocessor as SP
from pyimagesearch.preprocessing import meanpreprocessor as MP
from pyimagesearch.callbacks import epochcheckpoint as ECP
from pyimagesearch.callbacks import trainingmonitor as TM
from pyimagesearch.io import hdf5datasetgenerator as HDFG
from pyimagesearch.nn.conv import deepergooglenet as DGN
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import load_model
import keras.backend as K
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('-c','--checkpoints',required=True,
                help='path to output checkpoint directory')
ap.add_argument('-m','--model',type=str,
                help='path to *specific* model checkpoint to load')
ap.add_argument('-s','--start_epoch',type = int,default=0,
                help='epoch to restart training at')
args=vars(ap.parse_args())

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=18, zoom_range=0.15,
     width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
     horizontal_flip=True, fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SP.SimplePreprocessor(64,64)
mp = MP.MeanPreprocessor(means['R'],means['G'],means['B'])
iap = ITA.ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDFG.HDF5DatasetGenerator(config.TRAIN_HDF5,64,aug = aug,
preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)
valGen = HDFG.HDF5DatasetGenerator(config.VAL_HDF5,64,
preprocessors=[sp, mp, iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
    print("[INFO] compiling model....")
    model = DGN.DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
    #opt = Adam(1e-3)
	
    #opt = SGD(lr=1e-2,momentum=0.9)#Experiment #1
	#python train.py --checkpoints output/checkpoints
	
    #opt = SGD(lr=1e-3,momentum=0.9)#Experiment #2
	#python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_25.hdf5 --start_epoch 25
	
    opt = SGD(lr=1e-4,momentum=0.9)#Experiment #2
	#python train.py --checkpoints output/checkpoints --model output/checkpoints/epoch_35.hdf5 --start_epoch 35

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-5)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
callbacks = [
    ECP.EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
    TM.TrainingMonitor(config.FIG_PATH, jsonPath=config.JSON_PATH,startAt=args["start_epoch"])]

# train the network
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 64,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 64,
    epochs=30,
    max_queue_size=64 * 2,
    callbacks=callbacks, 
    verbose=1)

# close the databases
trainGen.close()
valGen.close()