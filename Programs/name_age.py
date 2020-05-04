# USAGE
# python3 test.py --encodings encodings.pickle --face "path to face_detector model" 
# --age "path to age_detector model" --image "path to images"

# import the necessary packages
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


def detect_faces():
    print("\t[INFO] computing face detections...")
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

            # extract the ROI of the face and then construct a blob from
            # *only* the face ROI
            face = image[startY:endY, startX:endX]
            faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                (78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False)
            
            age, ageConfidence =  age_detect(faceBlob)
            name = face_recongize(c)
            c = c + 1
            text1 = "{}".format(name)
            text2 = "Age:{}".format(age)
            text3 = "Prediction:{:.2f}%".format(ageConfidence * 100)
            
            y1 = endY + 20 if endY + 20 > 20 else endY - 20
            y2 = y1 + 20
            y3 = y2 + 20

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, text1, (startX, y1), cv2.FONT_HERSHEY_COMPLEX, 0.55, (255, 255, 0), 2)
            cv2.putText(image, text2, (startX, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)
            cv2.putText(image, text3, (startX, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 2)


def face_recongize(c):
    print("\t[INFO] recongizing face(s)...")
    # print(encoding.shape)
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

def age_detect(faceBlob):
    print("\t[INFO] detecting age(s)...")
    ageNet.setInput(faceBlob)
    preds = ageNet.forward()
    i = preds[0].argmax()
    age = AGE_BUCKETS[i]
    ageConfidence = preds[0][i]

    return age, ageConfidence


imagePaths = list(paths.list_images(args["image"]))

for (k, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(k + 1, len(imagePaths)))

    # load the input image and construct an input blob for the image
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

    encodings = face_recognition.face_encodings(rgb, boxes)
    # # pass the blob through the network and obtain the face detections
    # print("\t[INFO] computing face detections...")
    # faceNet.setInput(blob)
    # detections = faceNet.forward()
    
    detect_faces()

    print("\t[INFO] saving image...")
    text = "AgeFace{}.png".format(k+1)
    path = os.path.sep.join(["Output", text])
    cv2.imwrite(path, image)

# cv2.imshow("Image", image)
# cv2.waitKey(0)