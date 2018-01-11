import numpy as np
import cv2
import ccv
import cropper
import filefinder
import lbp
import tester

# define a main function
def main():
    #training = np.loadtxt("out.txt")
    training = filefinder.getTrainingData()        
    
    trainingdata = training[:,range(0,128)].astype(np.float32)
    responses = np.array(training[:,128]).reshape(1303,1).astype(np.float32)
    
    knn = cv2.ml.KNearest_create()
    knn.train(trainingdata, cv2.ml.ROW_SAMPLE, responses)

    tester.testme(knn)
    test = 5
    print("bitti")
if __name__ == "__main__":
    main()

cv2.waitKey(0)
cv2.destroyAllWindows();