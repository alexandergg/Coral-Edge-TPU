import cv2
import numpy as np

THRESHOLD = 600
PINK_COLOR = [[160,31,212],[180,49,288]]

def get_pink_bounding_box(image, bounding_box):
    box = np.array(bounding_box, dtype=np.int32)
    roi_height = box[3] - int((box[3]-box[1])/2) 
    roi = image[box[1]:roi_height, box[0]:box[2]]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array(PINK_COLOR[0])
    upper = np.array(PINK_COLOR[1])
    mask = cv2.inRange(hsv, lower, upper)
    return len(mask[mask>0]) > THRESHOLD