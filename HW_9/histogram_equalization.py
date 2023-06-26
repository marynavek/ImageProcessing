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


def cont_stretch(im, levels):
    im_out = np.zeros((im.shape[0],im.shape[1]), dtype=np.uint8)
    a, b = 0, levels-1
    c, d = im.min(), im.max()
    
    h, w = im.shape
    im_out[0:h, 0:w] = (im[0:h, 0:w] - c)*((b - a)/(d-c)) + a
    return im_out

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/reduced_version.jpg"
    
    image = cv2.imread(image_path, 0)

    plt.imshow(image, cmap='gray')
    plt.show()

    plt.hist(image.ravel(),256,[0,256])
    plt.show()

    transform = histogram_equalization(image)
    plt.imshow(transform, cmap='gray')
    plt.show()

    plt.hist(transform.ravel(),256,[0,256])
    plt.show()
   
    transform = cont_stretch(image, 500)
    plt.imshow(transform, cmap='gray')
    plt.show()

    plt.hist(transform.ravel(),256,[0,256])
    plt.show()

    transform = cont_stretch(image, 350)
    plt.imshow(transform, cmap='gray')
    plt.show()

    plt.hist(transform.ravel(),256,[0,256])
    plt.show()

    transform = cont_stretch(image, 150)
    plt.imshow(transform, cmap='gray')
    plt.show()

    plt.hist(transform.ravel(),256,[0,256])
    plt.show()