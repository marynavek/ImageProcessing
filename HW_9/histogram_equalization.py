import cv2
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization(image):
    histogram_array = np.bincount(image.flatten(), minlength=256)

    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels

    christogram_array = np.cumsum(histogram_array)

    transform_map = np.floor(255 * christogram_array).astype(np.uint8)

    img_list = list(image.flatten())

    eq_img_list = [transform_map[i] for i in img_list]

    eq_img_array = np.reshape(np.asarray(eq_img_list), image.shape)

    return eq_img_array


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/natual_im_1.jpg"
    
    image = cv2.imread(image_path, 0)
   
    transform = histogram_equalization(image)
    plt.imshow(transform, cmap='gray')
    plt.show()