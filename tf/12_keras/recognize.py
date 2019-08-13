import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.models import load_model
import argparse

img_rows, img_cols = 28, 28

# region of interest
def findRoi(frame, thresValue):
    rois = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.dilate(gray,None,iterations=2)
    gray2 = cv2.erode(gray2,None,iterations=2)
    edges = cv2.absdiff(gray,gray2)
    x = cv2.Sobel(edges,cv2.CV_16S,1,0)
    y = cv2.Sobel(edges,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
    ret, ddst = cv2.threshold(dst,thresValue,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(ddst,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 20:
            rois.append((x,y,w,h))
    return rois, edges


def findDigit(roi, thresValue):
    ret, th = cv2.threshold(roi, thresValue, 255, cv2.THRESH_BINARY)
    th = cv2.resize(th,(img_rows, img_cols))
    image = th.reshape(1, img_rows, img_cols, 1)
    pred_arr = model.predict(image)
    pred_num = pred_arr.argmax()

    return pred_num, th

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                  help="path to the image with handwriting numbers")
ap.add_argument("-d", "--direction", required=False, default="h",
                  help="show the comparison images 'h'orizontally or 'v'ertically")
args = vars(ap.parse_args())

print("Loading pre-trained model ...")
model = load_model("keras_cnn.model")

frame = cv2.imread(args["image"])
rois, edges = findRoi(frame, 50)

digits = np.zeros(shape=(img_rows, img_cols))
for r in rois:
    x, y, w, h = r
    digit, image = findDigit(edges[y:y+h,x:x+w], 50)
    # stack in a row
    digits = np.concatenate((digits, image), axis=1)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (153,153,0), 2)
    cv2.putText(frame, str(digit), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,0,255), 2)

newEdges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

if args["direction"] == "v":
    newFrame = np.vstack((frame,newEdges))
else:
    newFrame = np.hstack((frame,newEdges))

cv2.imshow('frame', newFrame)
cv2.imshow('digits(28 x 28)', digits)
key = cv2.waitKey(0) & 0xff

cv2.destroyAllWindows()
