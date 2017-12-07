# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys
import os
import glob
sys.path.append('/bigdrive/project/yolo-9000/darknet/python/')

import darknet as dn
import cv2
import joblib

net = dn.load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
meta = dn.load_meta("cfg/coco.data")


def detect_and_save(filename, outname):
    r = dn.detect(net, meta, filename, thresh=.15)
    img = cv2.imread(filename)
    for detection in r:
        label, probs, (x, y, w, h) = detection
        x = int(x - w/2)
        y = int(y - h/2)
        w = int(w)
        h = int(h)
        font_size = img.shape[0]/500
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(img, "{0}: {1}".format(label, round(probs, 3)) ,
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    cv2.imwrite(outname, img)
    return r

# still images
#detect_and_save("data/pacificrim.jpg", "data/detected_pacificrim.jpg")
#detect_and_save("data/kong.jpg", "data/detected_kong.jpg")

boxes = []
for filename in sorted(glob.glob("temp_photos/*")):
        outname = os.path.join("detected/"+os.path.basename(filename))
        box = detect_and_save(filename, outname)
        boxes.append(box)
joblib.dump(boxes, "detections.joblib")


