import math
import string
from multiprocessing import Process, Manager

import matplotlib.image as im
import numpy as np
from matplotlib import pyplot as plt

from numpy import ndarray


class borderDetector:

    def __init__(self, imgPath: string, sigmas: ndarray) -> None:
        """
        'borderDetector' is a class that performs the Laplacian of Gaussian and Gaussian's filter,
        given image with different Sigma values. The image is preprocessed (if occur) in order to
        transform it into a grey scale. It builds and convolve the image with different filter
        (built through the sigma value), in the end show for each sigma the convolved(filtered)
        image and the relative kernel(filter).
        In order to improve the execution of this script, it's worth use a multiprocessing approach
        thus we can parallelize the execution of different convolve-tasks.
        Each convolve-task is made by a different process (may increase the memory usage)
        :param imgPath: Image's path
        :param sigmas: array of different sigma values
        """
        self.__path = imgPath
        self.__sigmas = sigmas

        self.__source = None

    def detect(self) -> None:
        """
        Apply Laplacian of Gaussian
        :return:
        """
        # Retrieve the number of sigma values (perform convolution for each value)
        length = len(self.__sigmas)

        # empty array
        if length == 0:
            return

        # Retrieve a matrix-base image
        image = im.imread(self.__path)
        self.__source = np.copy(image)

        # We expect a grey scale image, but it's also ok a tensor (R,G,B matrices)
        if image.ndim != 2 and image.ndim != 3:
            return

        # If the image is a rgb based, we have to transform in grey scale image
        if image.ndim == 3:
            image = self.__rgbToGrey(image=image)

        # Dictionary where each process put the results (filtered image and kernel) inside
        results = Manager().dict()
        worker = np.empty(shape=len(self.__sigmas), dtype=Process)

        for i in range(0, length):
            # For each Sigma value, start the convolution
            worker[i] = Process(target=self.detector, args=(results, i, image, self.__sigmas[i]))
            worker[i].start()

        for i in range(0, length):
            worker[i].join()

        # 2 row , length columns
        fig, axs = plt.subplots(2, length)
        for i in range(0, length):
            axs[0, i].set_title('Convolved image S: ' + str(self.__sigmas[i]))
            axs[0, i].imshow(results[i][0], cmap="gray")
            axs[1, i].set_title('Kernel')
            axs[1, i].imshow(results[i][1])
        plt.show()
        return

    @staticmethod
    def __buildKernel(dim: int, sigma: float, krnl) -> ndarray:
        """
        Build the filter (in this case Laplacian of Gaussian) given kernel, dimensions and sigma
        :param dim: dimensions of filter
        :param sigma: sigma value used into LoG formula
        :param krnl: Kernel function
        :return: matrix that represent the filter
        """
        large = dim // 2  # "radius" of filter
        # Define an array between (-large to large) with size "dim"
        linSp = np.linspace(-large, large + 1, dim)
        # Create a matrix a square base (like a classical cartesian plan but in 2 dimension)
        X, Y = np.meshgrid(linSp, linSp)
        # Apply the Laplacian of Gaussian's formula
        return krnl(x=X, y=Y, sigma=sigma)

    @staticmethod
    def __loG(x: float, y: float, sigma: float) -> ndarray:
        # Laplacian of Gaussian 's formula
        s2 = np.power(sigma, 2)
        sub1 = -(np.power(x, 2) + np.power(y, 2)) / (2 * s2)
        return -(1 / math.pi * np.power(s2, 2)) * (1 + sub1) * np.exp(sub1)

    def detector(self, imgDict: dict, index: int, image: ndarray, sigma: float) -> None:
        """
        Given an image and sigma value apply the convolution and put the result in the dictionary
        :param imgDict: dictionary the put the result
        :param index: position of the dictionary where put the result
        :param image: image source where apply the convolution
        :param sigma: sigma value (used to build the filter)
        :return: is a function used in another process, no result returned
        """

        # Define the dimension of filter given a sigma value
        dim = 2 * int(4 * sigma + 0.5) + 1

        # Build the kernel (LoG)
        kernel = self.__buildKernel(dim=dim, sigma=sigma, krnl=self.__loG)
        # Convolve the image with the kernel, next we perform a sort of rescaling
        # we transform the float value in unsigned int (lie on 8 bytes)
        result = self.__convolution(image=image, kernel=kernel)

        imgDict[index] = [result.astype(np.uint8),kernel]
        return

    @staticmethod
    def __gaussian(x, y, sigma):
        s2 = 2 * np.power(sigma, 2)
        return 1 / (math.pi * s2) * np.exp(-(np.power(x, 2) + np.power(y, 2)) / s2)

    @staticmethod
    def __rgbToGrey(image: ndarray) -> ndarray:
        rgbConst = np.array([0.2989, 0.5870, 0.1140])
        image = np.array(
            rgbConst[0] * image[:, :, 0] + rgbConst[1] * image[:, :, 1] + rgbConst[2] * image[:, :, 2])

        return image

    @staticmethod
    def __convolution(image: ndarray, kernel: ndarray) -> ndarray:
        # Kernel dimensions, oss the filter is a square, we need just one side
        dimKer = kernel.shape[0]
        large = dimKer // 2  # "radius" of filter

        # We add 0-values around the image in order to apply the filter foreach pixel starting in (0,0) position
        # oss (0,0) position in the image becomes (large,larger) position in the padded image
        image = np.pad(image, pad_width=((large, large), (large, large)), mode='constant', constant_values=0) \
            .astype(np.float32)

        # new image
        result = np.zeros(image.shape)

        # Sliding the kernel along the source by pixel per pixel
        for h in range(large, image.shape[0] - large):
            for w in range(large, image.shape[1] - large):
                # 'square' represents a piece of image(source) with the same dimensions of a filter
                # We multiply the filter and the square than sum all values,this return a scalar.
                square = image[h - large:h - large + dimKer, w - large:w - large + dimKer]
                # pixel result
                result[h, w] = np.sum(np.multiply(square, kernel))

        # Return the convolved image
        return result[large:-large, large:-large]

if __name__ == '__main__':

    bd = borderDetector(imgPath="/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg", sigmas=np.array([0.73, 0.84]))
    bd.detect()