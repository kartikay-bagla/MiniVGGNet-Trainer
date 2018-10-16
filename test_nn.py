import numpy as np
import cv2
import time
import imutils
from imutils.video import VideoStream
from imutils import paths
import os
from keras.models import load_model
from helper import *
import tensorflow as tf
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required= True,
                help= "input video to test network on")
ap.add_argument("-m", "--model", required= True,
                help= "path to model used to test")
ap.add_argument("-o", "--output", required= True,
                help= "output video with classes mentioned")
ap.add_argument("-c", "--classes", 
                help= "Path to pickle file with list of classes indexed correctly for labelling")
args = vars(ap.parse_args())

with tf.device('/cpu:0'):
    model = load_model(args["model"])

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

preprocessors = [aap, iap]

vs = cv2.VideoCapture(args["input"])

if args["classes"] is None:
    classes = ['Cross', 'Nothing', 'Rectangle']
else:
    with open(args["classes"], "rb") as f:
        classes = pickle.load(f)
time.sleep(2.0)
counter = 0
times = []

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(args["output"], -1, 20.0, (600, 337))

while True:
    counter += 1
    start = time.time()
    frame = vs.read()
    frame = frame[1]
    if frame is None:
        break
    frame_ori = frame[:]
    frame = frame / 255.0
    for preprocessor in preprocessors:
        frame = preprocessor.preprocess(frame)

    pred = model.predict(np.array([frame]))
    pred = classes[np.argmax(pred)]

    stop = time.time()
    frame_ori = imutils.resize(frame_ori, width=600)

    times.append(stop-start)
    print(stop-start)
    cv2.putText(frame_ori, "{}".format(pred), (frame.shape[0]//2, frame.shape[1]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Frame", frame_ori)
    out.write(frame_ori)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

print("Average", np.average(times))

vs.release()
out.release()
cv2.destroyAllWindows()