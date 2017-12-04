import glob

import cv2

from face_extraction import process_face

for idx, frame in enumerate(sorted(glob.glob("data/*.jpg"))):
    img = cv2.imread(frame)
    process_face(img, output="test/%4s.jpg" % idx)
