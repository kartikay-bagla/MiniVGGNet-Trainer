from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import argparse

from helper import *
from minivggnet import *

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True,
                help= "path to dataset")
ap.add_argument("-m", "--model", required= True,
                help= "path to save model")
args = vars(ap.parse_args())

print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader([aap, iap])

data, labels = sdl.load(imagePaths, verbose= 500)
data = data.astype("float") / 255.0
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size= 0.25, random_state= 42)

print("[INFO] Creating model...")
aug = ImageDataGenerator(rotation_range= 30, width_shift_range= 0.1, height_shift_range= 0.1, shear_range= 0.2, 
                         zoom_range= 0.2, horizontal_flip= True, fill_mode= "nearest")

opt= SGD(lr= 0.05)
model = MiniVGGNet.build(64, 64, 3, len(classNames))
model.compile(optimizer= opt, loss= "categorical_crossentropy", metrics= ["accuracy"])

callbacks = [ModelCheckpoint("detector.model", verbose= 1, save_best_only=True, monitor="val_acc", mode= "max")]

print("[INFO] Beginning training...")
H = model.fit_generator(aug.flow(X_train, y_train, batch_size= 32), validation_data=(X_test, y_test), 
                        steps_per_epoch= len(X_train) // 32, epochs= 100, verbose= 1, callbacks= callbacks)

print("[INFO] Training finished...")