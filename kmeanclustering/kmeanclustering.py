import numpy as np
import cv2
import ccv
import cropper

img_o = cv2.imread('6.jpg')
a = cropper.cropImage(img_o)


cv2.imshow("test", a)
cv2.waitKey(0)
cv2.destroyAllWindows();

