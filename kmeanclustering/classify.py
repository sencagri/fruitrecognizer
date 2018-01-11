import numpy as np
import cv2
import ccv
import cropper
import filefinder
import lbp

class classifier:
    def __init__(self, _trainingdata, _responses):
        self.trainingdata = _trainingdata
        self.responses = _responses

    def findClass(self, newcomer):
        self.distances = np.copy(self.responses)

        for x,_ in enumerate(self.trainingdata):
            d = self.get_distance(newcomer, self.trainingdata[x])
            self.distances[x] = d
        
        self.distances = np.array(self.distances)
        res = np.concatenate((self.distances, self.responses), axis = 1)
        ind = np.lexsort([res[:,0]])
        res = res[ind]
        # get closest distance 
        dists = res[np.arange(1,6),:]
        result = np.zeros((15,1))
        for x,y in dists:
            result[int(y-1)] += 1
        mm = np.argmax(result) + 1

        return mm    

    def get_distance(self, p1, p2):
        p1 = p1.reshape((128,1))
        p2 = p2.reshape((128,1))
        psize = p1.size
        tdist = 0
        for i,_ in enumerate(p1):
            pa = p1[i]
            pb = p2[i]
            tdist += (pa - pb)**2
        tdist = tdist ** (1/2)
        return tdist
