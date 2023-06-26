import math
import numpy as np


class ButterworthFilter():

    def __init__(self):
        pass

    def __distance(self, point1,point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def FPLP(self, D0,imgShape,n):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1/(1+(self.__distance((y,x),center)/D0)**(2*n))
        return base

    def FPHP(self, D0,imgShape,n):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                base[y,x] = 1-1/(1+(self.__distance((y,x),center)/D0)**(2*n))
        return base