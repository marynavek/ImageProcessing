import cv2
import numpy as np
import matplotlib.pyplot as plt


def kirsch_filter(image, threshold):
    x,y = image.shape
    list=[]
    kirsch = np.zeros((x,y))
    for i in range(2,x-1):
        for j in range(2,y-1):
            d1 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d2 = np.square((-3) * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d3 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d4 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d5 = np.square(5 * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d6 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d7 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] - 
                  3 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d8 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            
            list=[d1, d2, d3, d4, d5, d6, d7, d8]
            kirsch[i,j]= int(np.sqrt(max(list)))
                         
    for i in range(x):
        for j in range(y):
            if kirsch[i,j]>255*threshold:
                kirsch[i,j]=255
            else:
                kirsch[i,j]=0
    return kirsch


def sobel_filter(image, threshold):
    #define horizontal and Vertical sobel kernels
    Gx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    Gy = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

    x,y  = image.shape
    sobel = np.zeros(shape=(x,y))

    for i in range(x - 2):
        for j in range(y - 2):
            gx = np.sum(np.multiply(Gx, image[i:i + 3, j:j + 3]))  # x direction
            gy = np.sum(np.multiply(Gy, image[i:i + 3, j:j + 3]))  # y direction
            sobel[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"
    
    for i in range(x):
        for j in range(y):
            if sobel[i,j]>255*threshold:
                sobel[i,j]=255
            else:
                sobel[i,j]=0

    return sobel

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"
    
    image = cv2.imread(image_path, 0)
   
    transform = sobel_filter(image, 0.2)
    plt.imshow(transform, cmap='gray')
    plt.show()