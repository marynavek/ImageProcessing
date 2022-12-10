import math
import numpy as np


class IdealFilter:

    def __init__(self):
        pass

    def __distance(self, point1,point2):
        return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

    def low_pass_filter(self, D0,imgShape):
        base = np.zeros(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if self.__distance((y,x),center) < D0:
                    base[y,x] = 1
        return base

    def high_pass_filter(self, D0,imgShape):
        base = np.ones(imgShape[:2])
        rows, cols = imgShape[:2]
        center = (rows/2,cols/2)
        for x in range(cols):
            for y in range(rows):
                if self.__distance((y,x),center) < D0:
                    base[y,x] = 0
        return base
    