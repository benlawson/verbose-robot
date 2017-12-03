# import os

import cv2
import numpy as np

screen_width = 1536
screen_height = 864


# top left to top right
for x in range(1, 10): # 10 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(x* (screen_width/10))
        col = int(screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)
        cv2.imwrite("data/0{0}{1}.jpg".format(x,y), img)

# top right to bottom right
for x in range(2, 10): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(9* (screen_width/10))
        col = int(x * screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)
        cv2.imwrite("data/1{0}{1}.jpg".format(x,y), img)

#  bottom right to bottom left
for x in range(2, 10): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(screen_width - (x* (screen_width/10)))
        col = int(screen_height/10)
        col = int(9 * screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)
        cv2.imwrite("data/2{0}{1}.jpg".format(x,y), img)

# bottom right to middle right
for x in range(2, 6): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(screen_width/10)
        col = int(screen_height - (x * screen_height/10))
        cv2.circle(img,(row, col), 50, (0,255,0), -1)
        cv2.imwrite("data/3{0}{1}.jpg".format(x,y), img)

# bottom right to middle right
for x in range(2, 8): # 9 dot placements
    for y in range(30): # 30 frames per second
        img = np.zeros((screen_height, screen_width, 3)) #three channel image
        row = int(x* (screen_width/10))
        col = int(5*screen_height/10)
        cv2.circle(img,(row, col), 50, (0,255,0), -1)
        cv2.imwrite("data/4{0}{1}.jpg".format(x,y), img)

#os.popen('ffmpeg -pattern_type glob -framerate 30 -i "data/*.jpg" output.mp4 -y')
