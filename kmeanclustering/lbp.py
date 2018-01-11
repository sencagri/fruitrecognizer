from skimage.feature import local_binary_pattern
import cv2
import numpy as np

radius = 8
n_points = radius * 8

def get_lbp(img):
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("test2", img_g)
    lbpimg = local_binary_pattern(img_g,n_points,radius,'uniform')

    (hist, _) = np.histogram(lbpimg.ravel(),
			bins=np.arange(0, n_points+1),
			range=(0, n_points+2))

    hist = hist.astype("float")
    #hist /= (hist.sum())
    """
    test = 5


    cv2.imshow("test", lbpimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lbpres =  np.array(local_binary_pattern(img_g, n_points, radius)).flatten().astype(np.float32)
    hist = np.histogram(lbpimg,64)
    """
    return hist.astype(np.float32)
