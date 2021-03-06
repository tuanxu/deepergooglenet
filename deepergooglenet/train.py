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
ap.add_argument('-e','--experiment',type = int,default=None,
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

experiment = "e%d"%args["experiment"]
print(experiment)
experiment_epoch = config.EXPERIMENT_CONFIGS[experiment][0]
experiment_optimize = config.EXPERIMENT_CONFIGS[experiment][1]
experiment_learning_rate = config.EXPERIMENT_CONFIGS[experiment][2]
    
print(experiment_epoch)
print(experiment_optimize)
print(experiment_learning_rate)
opt = ""	
	
if args["model"] is None:
    print("[INFO] compiling model....")
    model = DGN.DeeperGoogLeNet.build(width=64, height=64, depth=3, classes=config.NUM_CLASSES, reg=0.0002)
    


    if(experiment[1] == 'Adam'):
        opt = Adam(experiment_learning_rate)#Experiment #2-1
    else:
        opt = SGD(lr=experiment_learning_rate,momentum=0.9)



    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
else:
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])
    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, experiment_learning_rate)
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
    epochs=experiment_epoch,
    max_queue_size=64 * 2,
    callbacks=callbacks, 
    verbose=1)

# close the databases
trainGen.close()
valGen.close()