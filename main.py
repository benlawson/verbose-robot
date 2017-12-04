import glob

import cv2
import joblib
import numpy as np

from face_extraction import process_face
from gaze_prediction import MyNet


net = MyNet()

faces = []
right_eyes = []
left_eyes = []
grids = []
frames = sorted(glob.glob("./data/*"))

face_buffer  = np.zeros((256, 3, 224, 224))
right_buffer = np.zeros((256, 3, 224, 224))
left_buffer  = np.zeros((256, 3, 224, 224))
grid_buffer  = np.zeros((256, 625, 1, 1))

for idx, frame in enumerate(frames):
    if idx > 256:
        break
    img = cv2.imread(frame)
    if type(img) == type(None):
        face         = np.zeros((224, 224, 3))
        right_eye    = np.zeros((224, 224, 3))
        left_eye     = np.zeros((224, 224, 3))
        grid         = np.zeros((625, 1, 1))
    else:
        face, right_eye, left_eye, grid = process_face(img)

    face_buffer[idx]  = face.T
    right_buffer[idx] = right_eye.T
    left_buffer[idx]  = left_eye.T
    grid_buffer[idx]  = grid.T.reshape(625, 1, 1)

predictions = net.predict(face_buffer, right_buffer, left_buffer, grid_buffer)
joblib.dump(predictions, "predictions.joblib")
