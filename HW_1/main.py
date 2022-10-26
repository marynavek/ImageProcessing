import cv2
import numpy as np
import argparse

def gray_transform (image, im_name):
    # get the height and width of the image
    height, width = image.shape
    
    # create new empty image array to store pixels after the negative transform
    new_image = np.empty((image.shape), np.float64)

    for i in range(height):
        for j in range(width):
            
            pixel = 255 - image[i,j]
            new_image[i, j, ...] = pixel

    # save the new image
    cv2.imwrite(im_name, new_image)

def rgb_transform (image, im_name):
    # get the height and width of the image
    height, width, channel = image.shape
    
    # create new empty image array to store pixels after the negative transform
    new_image = np.empty((image.shape), np.float64)

    for i in range(height):
        for j in range(width):
            
            pixel_r = 255 - image[i,j, 0]
            pixel_g = 255 - image[i,j, 1]
            pixel_b = 255 - image[i,j, 2]

            new_image[i, j, 0] = pixel_r
            new_image[i, j, 1] = pixel_g
            new_image[i, j, 2] = pixel_b

    # save the new image
    cv2.imwrite(im_name, new_image)

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--image_path', type=str, required=True, help='Path to the original image.')

if __name__ == "__main__":
    args = parser.parse_args()
    image_path = args.image_path

    # Read the image using cv2 library
    image = cv2.imread(image_path)

    # Check it the input image is grayscale or RGB
    # If image has less than 3 channels, the program only takes negative transform of the gray scale image
    # Else the program (1) takes negative transform of the original image, (2) converts the original image to the grayscale, 
    # (3) takes negative transform of the gray scale version
    if len(image.shape) < 3:
    
        gray_transform(image, "gray_transform.jpg")

    else:
        rgb_transform(image, "rgb_transform.jpg")

        # get a gray scale version of the RGB image
        gray_image_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # save gray scale image version
        cv2.imwrite("gray_version.jpg", gray_image_1)

        gray_transform(gray_image_1, "gray_transform.jpg")

