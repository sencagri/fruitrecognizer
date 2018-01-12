# -*- coding:utf-8 -*-
import sys
import numpy as np
import cv2

def cropImage(img, x_scale = 0.5, y_scale = 0.5):
    img_o = img
    img = cv2.resize(img_o, (0,0), fx=x_scale, fy=y_scale,interpolation=cv2.INTER_CUBIC)
    onlyresized = img
    forcropping = img
    return img , onlyresized

    img = cv2.GaussianBlur(img, (3,3),0)
    #img= img_o
    #cv2.imshow("original", img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    ret,label,center=cv2.kmeans(Z,K,None,criteria,2,cv2.KMEANS_PP_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    im_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    ######################################################
    #MORPHOLOGICAL OPERATIONS
    ######################################################
    kernel = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    closing = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    cropped = cv2.bitwise_and(forcropping, closing)
    #cv2.imshow("cropped", cropped)
    return cropped,onlyresized
