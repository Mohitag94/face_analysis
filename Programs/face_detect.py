from imutils import paths
import face_recognition
import cv2
import sys
import os

# put path to the images inplace of path
imagePaths = list(paths.list_images("path"))

print(imagePaths)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model="cnn")

    for (top, right, bottom, left) in boxes:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    
    text = "{}.png".format(i+1)
    OutputPath = os.path.sep.join(["Output", text])
    print(OutputPath)
    cv2.imwrite(OutputPath, image)
