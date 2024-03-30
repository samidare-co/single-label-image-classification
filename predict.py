#import necessary packages
from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os


# Parse command line arguments
ap = argparse.ArgumentParser()


ap.add_argument("-i", '--input', required = True, help="path to input image/text file of image paths")
args = vars(ap.parse_args()) 

filetype = mimetypes.guess_type(args["input"])[0]
# One image path from the input 
imagePaths = [args["input"]]

if "text/plain" == filetype:
    # Override and populate the image path with all the image paths (one per line) in the text file
    imagePaths = open(args["input"]).read().strip().split("\n")


print("[INFO] loading object detector...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

# loop on the image paths and make prediction on each 
for imagePath in imagePaths:
    image = load_img(imagePath, target_size = (224,224))
     # convert to numpy array and scale pixel intensities to the range [0, 1]
    image = img_to_array(image)/ 255.0
    # add a batch dimension
    image = np.expand_dims(image, axis = 0)

    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    #determine the class label with the largest predicted probability
    print(labelPreds)
    i = np.argmax(labelPreds, axis = 1)
    label = lb.classes_[i][0]

    image = cv2.imread(imagePath)
    #Resize the image so that it fits in the screen 
    image = imutils.resize(image, width = 600)
    (h,w) = image.shape[:2]

    #Scale the bounding box coordinates back to their original dimensions
    startX = int(startX *w)
    startY = int(startY *h)
    endX = int(endX *w)
    endY = int(endY*w)

    #draw the predicted bounding box and label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.rectangle(image,(startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", image)
    # wait for a key press and then close all open windows
    cv2.waitKey(0)