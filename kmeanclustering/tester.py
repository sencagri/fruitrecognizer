import numpy as np
import lbp
import cv2
import ccv
import cropper
import os 
import classify

def testme(knn):
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
    bad = 0
    good = 0
    g_good=  0
    g_bad = 0
    testresult = np.zeros((1,64))
    firsttime = True
    for classname in filenamesWithLabel[:,0]:
        # now we are in the classes so we will get half of the images to trains out classifier
        imgnames = os.listdir(classname)
        imgnames = np.array([(classname + "/" + x) for x in imgnames])
        # here we get the names of the files in the directory related to a fruit or vegetable
        counter = 1
        for imgpath in imgnames:
            img = cv2.imread(imgpath)
            img,onlyresized = cropper.cropImage(img)
            ccvresult = np.array(ccv.get_ccv(img), np.float32).reshape(1,64)
            lbpresult = np.array(lbp.get_lbp(img)).reshape(1,64)
            #newcomer = ccvresult
            newcomer = np.concatenate([ccvresult,lbpresult],axis = 1)
            #ret, result, neihgbours, dist = knn.findNearest(newcomer, k=5)
            result = knn.findClass(newcomer)
            
            if(label == result):
                good += 1
                g_good += 1
            else:
                bad += 1
                g_bad += 1
            #print(classname + " label : " + str(label) + " result : " + str(result) + "     result" + str(label==result))
        
        print(classname + " good:" + str(good) + " bad:" + str(bad) + " ratio: % " + format(100*good/(bad+good),'.2f'))
        good = 0
        bad = 0
        
        label += 1
       

    print("global good:" + str(g_good) + " global bad:" + str(g_bad) + "global success:" + format(100*g_good/(g_good+g_bad),'.2f'))
    return testresult
