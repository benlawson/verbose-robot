import joblib
import cv2

frames = joblib.load('./example.joblib')
for idx, frame in enumerate(frames):
    cv2.imwrite("./data/{:>4}.jpg".format(idx), frame)

