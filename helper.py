from keras.preprocessing.image import img_to_array
import imutils
import cv2
import numpy as np 
import os

class ImageToArrayPreprocessor:
    def __init__(self, data_format= None):
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format= self.data_format)
        #print(a.shape)
    
class AspectAwarePreprocessor:
    def __init__(self, width, height, inter= cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        """
        Returns an image with the aspect ratio preserved but some cropping of edges.
        Step 1: Resize along smaller dimension
        Step 2: Crop along largest dimension to obtain target width and height.
        """
        
        # Resize along smaller axis and calculate the deltas
        h, w = image.shape[:2]
        dW, dH = 0, 0
        
        if w < h:
            image = imutils.resize(image, width= self.width, inter= self.inter)
            dH = int((image.shape[0] - self.height) / 2.0)
        else:
            image = imutils.resize(image, height= self.height, inter= self.inter)
            dW = int((image.shape[1] - self.width) / 2.0)
            
        #Re-Grab width and height and use delta to crop image
        h, w = image.shape[:2]
        image = image[dH:h-dH, dW: w-dW]
        
        #Resizing the image again as above step leads to +-1 error in dimensions.
        image = cv2.resize(image, (self.width, self.height), interpolation= self.inter)
        
        return image

class SimpleDatasetLoader(object):
    
    def __init__(self, preprocessors = []):
        """
        Loads images contained in a folder with preprocessors applied.
        The preprocessor should be an object with a method preprocess which takes the image as input and returns the output image.
        """
        self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        """
        Loads the imagePaths specified, applies the preprocessors to it and returns the images and labels.
        Verbosity level affects the frequency of messages. 
        For 3000 images and verbosity level 500, 6 messages will be printed.
        And for level 10, 300 messages."""
        data = []
        labels = []

        for i, path in enumerate(imagePaths):
            image = cv2.imread(path)
            label = path.split(os.path.sep)[-2]
            
            if self.preprocessors is not []:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

        return (np.array(data), np.array(labels))

