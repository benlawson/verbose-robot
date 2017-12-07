import cv2
import numpy as np


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')

# Malisiewicz et al.

# this was implemented by  Adrian Rosebrock at PyImageSearch
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

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
    display = img[:] # make a copy
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grid = np.zeros(gray.shape) # make a copy
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 1:
        #FIXME
        print("More than one face")
        # just taking the biggest one!
        size_of_bounding_box = []
        for x, y, w, h in faces:
            size_of_bounding_box.append(w*h)
        size_of_bounding_box = np.array(size_of_bounding_box)
        faces = [faces[size_of_bounding_box.argmax()]]

    if len(faces) < 1:
        print("Less than one face")
        #FIXME
        face = np.zeros((224, 224))
        print("No eyes!")
        right_eye = np.zeros((224, 224))
        left_eye = np.zeros((224, 224))

    # loop over faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = display[y:y+h, x:x+w]
        grid[y:y+h, x:x+w] = 1
        face = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(face_gray)
        labels = ['right', 'left']

        if len(eyes) > 2:
            print("More than two eyes")
            size_of_bounding_box = []
            for ex, ey, ew, eh in eyes:
                size_of_bounding_box.append(ew*eh)
            size_of_bounding_box = np.array(size_of_bounding_box)
            biggest = size_of_bounding_box.argmax()

            # new eyes will contain the two biggest boxes
            new_eyes = [eyes[biggest]]
            size_of_bounding_box[biggest] = 0
            biggest = size_of_bounding_box.argmax()
            new_eyes.append( eyes[biggest])
            eyes = new_eyes

        elif len(eyes) == 1:
            #FIXME
            print("Less than two eyes")
            left_eye = np.zeros((224, 224))
        elif len(eyes) < 1:
            #FIXME
            print("No eyes!")
            right_eye = np.zeros((224, 224))
            left_eye = np.zeros((224, 224))

        # loop over eye detections
        for (label, (ex,ey,ew,eh)) in zip(labels, sorted(eyes, key=lambda d: d[0])):

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
        grid = cv2.resize(grid, (25, 25))
    return face, right_eye, left_eye, grid


# test = "/bigdrive/gaze/data/01036/frames/00151.jpg"
# test_image = cv2.imread(test)
# process_face(test_image)
