import numpy as np
import cv2
import ccv
import cropper
import filefinder
import lbp
import tester
import classify

# define a main function
def main():
    training = np.loadtxt("out.txt")
    #training = filefinder.getTrainingData()        
    
    trainingdata = training[:,range(0,128)].astype(np.float32)
    size = len(trainingdata)
    responses = np.array(training[:,128]).reshape(size,1).astype(np.float32)
    
    cc = classify.classifier(trainingdata, responses)
    
    #tester.testme(knn)
    tester.testme(cc)
    print("end of program")

if __name__ == "__main__":
    main()
