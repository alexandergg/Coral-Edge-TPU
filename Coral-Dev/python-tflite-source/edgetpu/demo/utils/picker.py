import cv2
import numpy as np
import argparse

image_src = None   # global ;(
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_src[y,x]
        print(y,x)
        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(pixel, lower, upper)
        image_mask = cv2.inRange(image_src,lower,upper)
        cv2.imshow("mask",image_mask)


def main(image_path):
    import sys
    global image_src, pixel # so we can use it in mouse callback

    image_src = cv2.imread(image_path)  # pick.py my.png
    image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2HSV)
    if image_src is None:
        print ("the image read is None............")
        return

    cv2.namedWindow('bgr')
    cv2.setMouseCallback('bgr', pick_color)
    cv2.imshow("bgr",image_src)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Color Picker for opencv image')
    parser.add_argument('image_path', help='image path')
    args = parser.parse_args()
    main(args.image_path)