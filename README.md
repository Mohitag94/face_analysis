# face_analysis
Face Detect and Recognition, Age and Emotion Detection

Python3.6.9 used.
# Required Python Packages:
```
openCV
face_recognition
keras
tensorflow
```

# Link to datasets:

[LFW](http://vis-www.cs.umass.edu/lfw/)
[Adience](https://talhassner.github.io/home/projects/Adience/Adience-data.html#agegender)
[FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

# Uages:
### face_detect: 
```
python3 face_detect (put the path to images inplace of "path")
```
### recognize_faces: 
```
python3 recognize_faces.py --encodings "path to encodings.pickle" --image "path to images to recognize"
```
### age_detect: 
```
python3 age_detect.py --image "path to images"  --face "path to face_detector model" --age "path to age_detector model"
```
### emotion_detect:
```
python3 emotion.py --emotion "path to emotion_detector model" --image "path to images for predictions"
```
### name_age:
```
python3 test.py --encodings encodings.pickle --face "path to face_detector model" --age "path to age_detector model" --image "path to images"
```
### EmotionAgeFace: 
```
python3 EmotionAgeFace.py --encodings "path to encodings.pickle file" --face "path to face_detector model" --age "path to age_detector model" --emotion "path to emotion_detector model" --image "path to images"
```
