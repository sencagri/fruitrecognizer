import numpy as np
import lbp
import cv2
import ccv
import cropper
import os 

def getTrainingData():
    path = "dataset/"
    # get all directories
    filenames = os.listdir(path)

    # add dataset/ to all path
    filenames = np.array([(path + x) for x in filenames]).reshape(15,1)

    # find number of classes
    classNumber = len(filenames)
    numerator = np.arange(1,classNumber+1, 1).reshape(15,1)

    # merge numberClass and paths
    filenamesWithLabel = np.hstack((filenames, numerator))

    # create a list to hold training data
    label = 1
    training = np.zeros((1,64))
    firsttime = True
    for classname in filenamesWithLabel[:,0]:
        # now we are in the classes so we will get half of the images to trains out classifier
        imgnames = os.listdir(classname)
        imgnames = np.array([(classname + "/" + x) for x in imgnames])
        # here we get the names of the files in the directory related to a fruit or vegetable
        counter = 1
        for imgpath in imgnames:
            img = cv2.imread(imgpath)
            img = cropper.cropImage(img)[0]
            ccvresult = np.array(ccv.get_ccv(img), np.float32).reshape(1,64)
            lbpresult = np.array(lbp.get_lbp(img)).reshape(1,64)
            labelarray =  np.array(label).reshape(1,1)
            result = np.concatenate([ccvresult,lbpresult],axis = 1)
            result = np.concatenate([result,labelarray],axis = 1)
            #result = np.concatenate([ccvresult,labelarray],axis = 1)
            if(firsttime):
                training = result
                firsttime = False
            else:
                training = np.vstack((training,result))
            
            counter += 1
            print(classname + " " + str(counter))

        
        label += 1
        print(str(training.shape))
        np.savetxt("out.txt",training)

    return training
