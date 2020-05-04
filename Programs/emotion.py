# USAGE
# python3 fae_test.py --emotion "path to emotion_detector model" --image "path to images for predictions"

# import the necessary packages
from imutils import paths
from numpy import loadtxt
from keras.models import load_model
import numpy as np
import face_recognition
import argparse
import pickle
import sys
import os
import cv2

# # construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-em", "--emotion", required=True,
	help="path to emotion detector model directory")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# define the list of emotion for prediction
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

# load the emotion model
print("[INFO] loading emotion model")
emotionPath = os.path.sep.join([args["emotion"], "emotion.hdf5"])
print(emotionPath)
model = load_model(emotionPath)

def detect_faces(boxes):
    for (i, box) in enumerate(boxes):
        print("\t[INFO] computing face detections...")
        
        (top, right, bottom, left) = box

        face = image[top:bottom, left:right]

        emotion = emotion_detector(face)

        text = "Emotion: {}".format(emotion)
        
        y1 = top - 15 if top + 15 > 15 else top - 15

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.putText(image, text, (left, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)



def emotion_detector(face):
    print("\t[INFO] detecting emotion...")
    face_image = cv2.resize(face, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    predicted_class = np.argmax(model.predict(face_image))

    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    predicted_label = label_map[predicted_class]
            
    print(predicted_label)
    return predicted_label


imagePaths = list(paths.list_images(args["image"]))

for (k, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(k + 1, len(imagePaths)))

    # load the input image and construct an input blob for the image
    image = cv2.imread(imagePath)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    detect_faces(boxes)

    cv2.putText(image, "Faces:{}".format(len(boxes)), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

    print("\t[INFO] saving image...")
    text = "EmotionDetect{}.png".format(k+1)
    path = os.path.sep.join(["Output", text])
    cv2.imwrite(path, image)

# cv2.imshow("Image", image)
# cv2.waitKey(0)