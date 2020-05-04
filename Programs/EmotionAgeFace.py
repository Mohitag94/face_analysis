# USAGE
# python3 EmotionAgeFace.py --encodings "path to encodings.pickle file" --face "path to face_detector model" 
# --age "path to age_detector model" --emotion "path to emotion_detector model" --image "path to images"

# import the necessary packages
from keras.models import load_model
from numpy import loadtxt
from imutils import paths
import numpy as np
import face_recognition
import argparse
import pickle
import sys
import os
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-f", "--face", required=True,
	help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
	help="path to age detector model directory")
ap.add_argument("-em", "--emotion", required=True,
	help="path to emotion detector model directory")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# define the list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the emtion model
print("[]INFO loading emotion model")
emotionPath = os.path.sep.join([args["emotion"], "emotion.hdf5"])
print(emotionPath)
model = load_model(emotionPath)

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

imagePaths = list(paths.list_images(args["image"]))
print(imagePaths)

# fumctions---->>>
def detect_faces():
    print("\t[INFO] computing face detections...")
    print("\t\t[INFO] faces : %s" % str(len(boxes)))
    faceNet.setInput(blob)
    detections = faceNet.forward()
    c = 0
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            print(startX, startY, endX, endY)

            # extract the ROI of the face and then construct a blob from
            # *only* the face ROI
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False)
            
            age, ageConfidence =  age_detect(faceBlob, c)
            name = face_recongize(c)
            emotion = emotion_detect(face, c)
            c = c + 1
            text_rect(age, ageConfidence, name, emotion, startX, startY, endX, endY)

def face_recongize(c):
    print("\t[INFO] recognizing face...%s" % str(c+1))
    matches = face_recognition.compare_faces(data["encodings"], encodings[c])
    # print(matches)
    name = "Unknown"
    # check to see if we have found a match
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
		
        name = max(counts, key=counts.get)

    return name

def age_detect(faceBlob, c):
    print("\t[INFO] detecting age...%s" % str(c+1))
    ageNet.setInput(faceBlob)
    preds = ageNet.forward()
    i = preds[0].argmax()
    age = AGE_BUCKETS[i]
    ageConfidence = preds[0][i]

    return age, ageConfidence

def emotion_detect(face, c):
    print("\t[INFO] detecting emotion...%s" % str(c+1))
    face_image = cv2.resize(face, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

    predicted_class = np.argmax(model.predict(face_image))

    label_map = dict((v,k) for k,v in emotion_dict.items()) 
    predicted_label = label_map[predicted_class]
            
    # print(predicted_label)
    return predicted_label

def text_rect(age, ageConfidence, name, emotion, startX, startY, endX, endY):
    text1 = "{}".format(name)
    text2 = "Age:{}".format(age)
    text3 = "Emotion:{}".format(emotion)
    text4 = "Prediction:{:.2f}%".format(ageConfidence * 100)
    
    y1 = endY + 20 if endY + 20 > 20 else endY - 20
    y2 = y1 + 20
    y3 = y2 + 20
    y4 = y3 + 20

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.putText(image, text1, (startX, y1), cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 255, 0), 2)
    cv2.putText(image, text2, (startX, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
    cv2.putText(image, text3, (startX, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
    cv2.putText(image, text4, (startX, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)

# calculating name, age, emotion for each image
for (k, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(k + 1, len(imagePaths)))

    # load the input image and construct an input blob for the image
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    print(boxes)
    encodings = face_recognition.face_encodings(rgb, boxes)
    # c = 0 
    detect_faces()

    text = "Face_Count : {}".format(len(boxes))
    cv2.putText(image, text, (10, 15), cv2.FONT_HERSHEY_COMPLEX, 0.55, (0, 0, 255), 2)

    print("\t[INFO] saving image...")
    text = "EmotionAgeFace{}.png".format(k+1)
    path = os.path.sep.join(["Outputs", text])
    cv2.imwrite(path, image)