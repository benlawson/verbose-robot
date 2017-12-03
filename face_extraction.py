import logging

import cv2

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.FileHandler('logs.log')
logger.setLevel(logging.DEBUG)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

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

def process_face(img, resize=True, output=False):
    '''
    takes an image and detects face/eyes
    @params:
        img (numpy.array): an opencv image (hopefully with a face in it!)
        resize (bool):  all crops are resized to 224x224 if True, default: True
        output (str, bool): if a string is provided, a copy of the image
                            detections will be saved at this location, default: False
    @returns
    face (numpy.array): roi of interest corresponding to a face
    right_eye (numpy.array): roi of interest corresponding to the right eye
    left_eye  (numpy.array): roi of interest corresponding to the left eye
    face_grid (numpy.array): a grid that is marked where the face is
    '''
    display = img[:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 1:
        logger.warning("More than one face")
    # loop over faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = display[y:y+h, x:x+w]
        face = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)
        #eyes = sorted(eyes, lambda d,e: d[0] < e[0],reverse=True)
        labels = ['right', 'left']

        # loop over eye detections
        for (label, (ex,ey,ew,eh)) in zip(labels, eyes):
            if len(eyes) > 2:
                logger.warning("More than two eyes")
            elif len(eyes) < 2:
                logger.warning("Less than two eyes")
            cv2.rectangle(face_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(face_color, label, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
            if label == 'right':
                right_eye = face_color[ey:ey+eh, ex:ex+ew]
            elif label == 'left':
                left_eye  = face_color[ey:ey+eh, ex:ex+ew]
    if output:
        cv2.imwrite(output, display)
    if resize:
        face = cv2.resize(face, (224, 224))
        right_eye = cv2.resize(right_eye, (224, 224))
        left_eye = cv2.resize(left_eye, (224, 224))
    return face, right_eye, left_eye, None


# test = "/bigdrive/gaze/data/01036/frames/00151.jpg"
# test_image = cv2.imread(test)
# process_face(test_image)
