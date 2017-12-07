import os
import glob

import cv2
import joblib
from rtree import index


def convert_chromebook(pred):
    x, y = pred
    screen_width = 1536 #pixel
    screen_height = 864 #pixel
    screen_physical_width =  7.9 # inch
    screen_physical_height =  5.5 # inch

    x = (x * 0.393701  *  screen_width / screen_physical_width) + screen_width/2.8
    y = -1* (y * 0.393701  *  screen_height / screen_physical_height) + screen_height/1.4
    return int(x), int(y)

def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2-x1)*(y2-y1)

### run yolo object detections
### save detections per frame as a list in a joblib file

detections = joblib.load("../my_yolo/detections.joblib")
predictions = joblib.load("../predictions.joblib")
frames = sorted(glob.glob("../my_yolo/temp_photos/*"))

for detection, prediction, frame in zip(detections, predictions, frames):
    outname = os.path.join("yolo_data", os.path.basename(frame))
    if len(detection) < 1:
        #FIXME
        # just plot point
        img = cv2.imread(frame)
        color = (255, 0, 0)
        pred_x, pred_y = convert_chromebook(prediction)
        cv2.circle(img,(pred_x, pred_y), 25, (255,0,0), -1)

        cv2.imwrite(outname, img)
        continue

    idx = index.Index() # create r tree
    for detect_num, d in enumerate(detection):
        label, probs, (x, y, w, h) = d
        xmin = int(x - w/2) # center to top left
        ymin = int(y - h/2) # center to top left
        xmax = int(xmin + w)
        ymax = int(ymin + h)
        #print(detect_num, (xmin, ymin, xmax, ymax))
        idx.insert(detect_num, (xmin, ymin, xmax, ymax))
    pred_x, pred_y = convert_chromebook(prediction)
    closest = list(idx.nearest((pred_x, pred_y), 3, objects=True))[0]

    query = [0, 0, 0, 0]
    xmin, ymin, xmax, ymax = closest.bounds
    if xmin < xmax:
        query[0] = xmin
        query[2] = xmax
    else:
        query[2] = xmin
        query[0] = xmax
    if ymin < ymax:
        query[1] = ymin
        query[3] = ymax
    else:
        query[1] = ymax
        query[3] = ymin


    overlaps = idx.intersection(query, objects=True)

    # if nearest neighbors overlap, take the smallest one
    smallest = 99999
    for overlap in overlaps:
        area = get_bbox_area(overlap.bbox)
        if area < smallest:
            closest = overlap
            smallest = area


    closest = overlap.id

    # plot bounding box
    img = cv2.imread(frame)
    for detect_num, d in enumerate(detection):
        if detect_num == closest:
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)
        font_size = img.shape[0]/750
        label, probs, (x, y, w, h) = d
        x = int(x - w/2) # center to top left
        y = int(y - h/2) # center to top left
        w = int(w)
        h = int(h)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.putText(img, "{0}".format(label, round(probs, 3)) ,
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

        color = (255, 0, 0)
        pred_x, pred_y = convert_chromebook(prediction)
        cv2.circle(img,(pred_x, pred_y), 25, (255,0,0), -1)
    cv2.imwrite(outname, img)


