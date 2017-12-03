import glob
import logging

import cv2

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.FileHandler('logs.log')
logger.setLevel(logging.DEBUG)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def crop_to_roi(image, x, y, width, height):
    ''' returns an roi correspoinding of a crop from the frame
    @params:
        image: numpy.array opencv image
        x: x coordinate
        y: y coordinate
        width: width of roi
        height: height of roi
    @returns:
        numpy.array (width, height)
    '''
    return image[y:y+height, x:x+width]

def process_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 1:
        logger.warning("More than one face")
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)
        eyes = sorted(lambda d: d[0], eyes, reverse=True)
        labels = ['right', 'left']
        for (label, (ex,ey,ew,eh)) in zip(labels, eyes):
            if len(eyes) > 2:
                logger.warning("More than two eyes")
            elif len(eyes) < 2:
                logger.warning("Less than two eyes")
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            #                 0.5, (255, 255, 255), 2)
            right_eye (face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imwrite("test.jpg", img)


# assume batch of 256 frames (batch size used downstream

faces = []
right_eyes = []
left_eyes = []
grids = []
frames = glob.glob("./data/*")
for frame in frames:
    face, right_eye, left_eye, grid = process_face(frame)
    faces.append(face)
    right_eyes.append(right_eye)
    left_eyes.append(left_eye)
    grids.append(grid)

test = "/bigdrive/gaze/data/01036/frames/00151.jpg"
test_image = cv2.imread(test)
process_face(test_image)
