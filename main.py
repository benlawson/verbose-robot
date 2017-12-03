import os
import json
import glob

import matplotlib
matplotlib.use('Agg')
import cv2
import caffe
import numpy as np

model = "../models/itracker_deploy.prototxt"
weights = "../models/snapshots/itracker_iter_92000.caffemodel"

class MyNet(object):
    def __init__(self):
        self.net = caffe.Net(model, weights, caffe.TEST)

    def predict(self, image_left, image_right, image_face, facegrid):
        # batch size is 256
        image_left = cv2.resize(image_left, (224, 224)).T
        image_left = np.array([image_left  for _ in range(256)])
        self.net.blobs["image_left"].data[...] = image_left

        image_right = cv2.resize(image_right, (224, 224)).T
        image_right = np.array([image_right for _ in range(256)])
        self.net.blobs["image_right"].data[...] = image_right

        image_face = cv2.resize(image_face, (224, 224)).T
        image_face = np.array([image_face for _ in range(256)])
        self.net.blobs["image_face"].data[...] = image_face

        facegrid = cv2.resize(facegrid, (25, 25)).T.reshape(625, 1, 1)
        facegrid = np.array([facegrid for _ in range(256)])
        self.net.blobs["facegrid"].data[...] = facegrid

        return self.net.forward()['fc3'].mean(axis=0)

    def load_json_data(self, path_to_file, num_of_frames=1):
        info = json.load(open(path_to_file))
        keys = ['X', 'Y','W', 'H']
        return [info[key][num_of_frames] for key in keys]



    def preprocess(self, path_to_subject, num_of_frames=1):
        frame_filenames = sorted(glob.glob(os.path.join(path_to_subject, "frames/*")))
        left_eye_info = self.load_json_data(os.path.join(path_to_subject, "appleLeftEye.json"), num_of_frames)
        right_eye_info = self.load_json_data(os.path.join(path_to_subject, "appleRightEye.json"),num_of_frames )
        face_info = self.load_json_data(os.path.join(path_to_subject, "appleFace.json"), num_of_frames)
        face_grid_info = self.load_json_data(os.path.join(path_to_subject, "faceGrid.json"),num_of_frames)



        for n in range(num_of_frames):
            frame_filename = frame_filenames[n]
            left_eye = left_eye_info
            right_eye = right_eye_info
            face = face_info
            face_grid = face_grid_info

            print(frame_filename)
            img = cv2.imread(frame_filename)
            left_eye_roi = self.get_roi(img, left_eye)
            right_eye_roi = self.get_roi(img, right_eye)
            face_roi = self.get_roi(img, face)
            face_grid_roi = self.get_roi(np.zeros((25,25)), face_grid)

        return left_eye_roi, right_eye_roi, face_roi, face_grid_roi


    def get_roi(self, img, roi):
        '''
        @params:
            img (np.array): an cv2 image
        @returns:
            (np.array) a roi from the image
        '''

        x, y, w, h = roi

        y1 = int(y)
        y2 = int(y1 + h)

        x1 = int(x)
        x2 = int(x1 + w)

        return img[y1:y2, x1:x2]


net = MyNet()
