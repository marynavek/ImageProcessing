import math
import numpy as np
from matplotlib import pyplot as plt
import cv2

def buildKernel(dim, sigma, kernel):
    large = dim // 2
    lin_space = np.linspace(-large, large+1, dim)
    X, Y = np.meshgrid(lin_space, lin_space)

    return kernel(x=X, y=Y, sigma=sigma)

def loG(x, y, sigma):
    s2 = np.power(sigma, 2)
    sub1 = -(np.power(x, 2) + np.power(y,2)) / (2 * s2)
    return -(1 / math.pi * np.power(s2, 2)) * (1 + sub1) * np.exp(sub1)

def convolution(image, kernel):
    dimKer = kernel.shape[0]
    large = dimKer // 2

    image = np.pad(image, pad_width=((large, large), (large, large)), mode = 'constant', constant_values=0).astype(np.float32)

    result = np.zeros(image.shape)

    for h in range(large, image.shape[0]-large):
        for w in range(large, image.shape[1]-large):
            square = image[h - large:h - large + dimKer, w - large:w - large+ dimKer]

            result[h,w] = np.sum(np.multiply(square, kernel))

    return result[large:-large, large:-large]

def detector(image, sigma):
    #sigma value definition
    dim = 2*int(4*sigma + 0.5) +1

    kernel = buildKernel(dim=dim, sigma=sigma, kernel=loG)

    result = convolution(image=image, kernel=kernel)

    return result


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_1.png"
    
    image = cv2.imread(image_path, 0)
   
    transform = detector(image, 0.75)
    plt.imshow(transform, cmap='gray')
    plt.show()