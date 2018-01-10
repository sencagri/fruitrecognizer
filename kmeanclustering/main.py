import numpy as np
import cv2
import ccv
import cropper
import filefinder
import os

# define a main function
def main():
    training = np.loadtxt("out_.txt")

    #training = filefinder.getTrainingData()        

    trainingdata = training[:,range(0,64)].astype(np.float32)
    responses = np.array(training[:,64]).reshape(666,1).astype(np.float32)
    knn = cv2.ml.KNearest_create()
    knn.train(trainingdata, cv2.ml.ROW_SAMPLE, responses)

    test = cv2.imread("dataset/plum/plum_151.jpg");
    test2 = cropper.cropImage(test)
    newcomer = np.array(ccv.get_ccv(test2), np.float32).reshape(1,64).astype(np.float32)
    ret, result, neihgbours, dist = knn.findNearest(newcomer, k=1)
    test = 5
if __name__ == "__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows();