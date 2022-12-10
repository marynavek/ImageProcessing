from butterworth_filter import ButterworthFilter
from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/shape_circle.png"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (32,32))
    plt.imshow(image, cmap='gray')
    plt.show()
    imgTrans = np.fft.fftshift(np.fft.fft2(image))
    filter = ButterworthFilter()
    imgLowPass = filter.FPLP(10,image.shape, 20)
    imgProcessLP = imgTrans*imgLowPass

    plt.imshow(imgLowPass, cmap='gray')
    plt.show()

    inversaLP = np.fft.ifftshift(imgProcessLP) 
    inversaLP = np.fft.ifft2(inversaLP)  
    inversaLP = np.abs(inversaLP)

    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(inversaLP, cmap='gray')
    plt.show()

    imgHighPass = filter.FPHP(50,image.shape, 20)
    imgProcessHighPass = imgTrans*imgHighPass
    inversaHighPass= np.fft.ifftshift(imgProcessHighPass) 
    inversaHighPass = np.fft.ifft2(inversaHighPass)  
    inversaHighPass = np.abs(inversaHighPass)