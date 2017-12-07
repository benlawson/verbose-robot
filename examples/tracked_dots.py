# import os

import cv2
import joblib
import numpy as np

screen_width = 1536 #pixel
screen_height = 864 #pixel

# screen_physical_width =  10.125 # inch
# screen_physical_height =  5.75  # inch

screen_physical_width =  7.9 # inch
screen_physical_height =  5.3 # inch



### code for ipad mini ###
#screen_physical_width =  5 # inch
#screen_physical_height =  8.4 # inch
#
#predictions = joblib.load("../predictions.joblib")
#ys, xs = zip(*predictions)
#
#ys = (np.array(ys) * 0.393701  *  screen_height / screen_physical_height) + screen_height/8
#xs = (np.array(xs) * 0.393701  *  screen_width / screen_physical_width)
#

### code for chromebook ###
predictions = joblib.load("../predictions.joblib")
ys, xs = zip(*predictions)

screen_physical_width =  7.9 # inch
screen_physical_height =  5.5 # inch

xs = (np.array(xs) * 0.393701  *  screen_width / screen_physical_width) + screen_width/2.8
ys = -1* (np.array(ys) * 0.393701  *  screen_height / screen_physical_height) + screen_height/1.4
#


# remove garabage points
# xs[np.where(xs == 621.34094)] = np.nan
# xs[np.where(xs == 621.34088)] = np.nan

# ys[np.where(ys == 16.699905)] = np.nan
# ys[np.where(ys == 16.699926)] = np.nan

ys = np.mean(ys.reshape(-1, 4), axis=1)
xs = np.mean(xs.reshape(-1, 4), axis=1)

# top left to top right
for x in range(1, 10): # 10 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(x* (screen_width/10))
        col = int(screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)

        track_row = xs[x-1]
        track_col = ys[x-1]
        cv2.circle(img,(track_row, track_col), 25, (255,0,0), -1)

        cv2.imwrite("data/0{0}{1}.jpg".format(x,y), img)

# top right to bottom right
for x in range(2, 10): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(9* (screen_width/10))
        col = int(x * screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)

        track_row = xs[x+7]
        track_col = ys[x+7]
        cv2.circle(img,(track_row, track_col), 25, (255,0,0), -1)

        cv2.imwrite("data/1{0}{1}.jpg".format(x,y), img)

#  bottom right to bottom left
for x in range(2, 10): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(screen_width - (x* (screen_width/10)))
        col = int(screen_height/10)
        col = int(9 * screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)

        track_row = xs[x+14]
        track_col = ys[x+14]
        cv2.circle(img,(track_row, track_col), 25, (255,0,0), -1)

        cv2.imwrite("data/2{0}{1}.jpg".format(x,y), img)

# bottom right to middle right
for x in range(2, 6): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(screen_width/10)
        col = int(screen_height - (x * screen_height/10))
        cv2.circle(img,(row, col), 50, (0,255,0), -1)

        track_row = xs[x+21]
        track_col = ys[x+21]

        cv2.circle(img,(track_row, track_col), 25, (255,0,0), -1)
        cv2.imwrite("data/3{0}{1}.jpg".format(x,y), img)

# bottom right to middle right
for x in range(2, 8): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(x* (screen_width/10))
        col = int(5*screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)

        track_row = xs[x+24]
        track_col = ys[x+24]

        cv2.circle(img,(track_row, track_col), 25, (255,0,0), -1)
        cv2.imwrite("data/4{0}{1}.jpg".format(x,y), img)

#os.popen('ffmpeg -pattern_type glob -framerate 30 -i "data/*.jpg" output.mp4 -y')
