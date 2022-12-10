import numpy as np


# def determine_difference(original_image, reconstructed_image):
#     difference = np.subtract(original_image, reconstructed_image)


def mse_between_two_images(original_image, reconstructed_image):
    # original_image = np.array(original_image)
    # reconstructed_image = np.array(reconstructed_image)
    differences = np.subtract(original_image, reconstructed_image)
    squared_differences = np.square(differences)
    return squared_differences.mean()

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"


    # do it for fft, dct, and walsh