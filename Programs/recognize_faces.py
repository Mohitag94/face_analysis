# USAGE
# python3 batch_image_recognize_faces.py --encodings "path to encodings.pickle" --image "path to images to recognize"

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import csv
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# print(args)

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["image"]))
# print(imagePaths)
k = 0
for (i, imagePath) in enumerate(imagePaths):
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(imagePath)
    # image = cv2.resize(img, (0, 0), None, .25, .25)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))
    print("\t[INFO] recognizing faces...")

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    # print("[INFO] recognizing faces from image {}".format(i+1))
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
        
        # update the list of names
        names.append(name)

    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # print(image)
    # saving the image 
    print("\t[INFO] saving recognized faces...")
    text = "reconizedFace{}.png".format(k+1)
    # print(k)
    # print(text)
    path = os.path.sep.join(["Outputs", text])
    cv2.imwrite(path, image)
    print("\t[INFO] done")
    k = k + 1
    # show the output image
    # cv2.imshow("Image", image)
    
# cv2.waitKey(0)
cv2.destroyAllWindows()