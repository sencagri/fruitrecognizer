from skimage.feature import local_binary_pattern
import cv2
import numpy as np

radius = 4
n_points = radius * 16

def get_lbp(img):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbpimg =  np.array(local_binary_pattern(img_g, n_points, radius)).flatten().astype(np.float32)
    hist = np.histogram(lbpimg,64)
    return hist[0].astype(np.float32)
