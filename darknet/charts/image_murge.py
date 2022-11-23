import numpy as np
import cv2

img1 = cv2.imread('v3/chart_1차(11.18 20:42).png')
img2 = cv2.imread('v3/chart_2차(11.19 11:00).png')
img3 = cv2.imread('v3/chart_3차(11.19 23:20).png')

def weight(x):
    pass

cv2.namedWindow('image')
cv2.createTrackbar('weight','image',0,100,weight)

while True:
    weight = cv2.getTrackbarPos('Weight','image')
    addWeight = cv2.addWeighted(img1,float(100-weight) * 0.01, img2, float(weight) * 0.01, 0)
    cv2.imshow('image', addWeight)

    if cv2.waitKey(1) &0xFF == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()