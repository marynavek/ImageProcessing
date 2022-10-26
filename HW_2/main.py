import numpy as np
import cv2, os


if __name__ == "__main__":
    print("hello")
    image_path = "/Users/marynavek/Projects/ImageProcessing/HW_2/lady_rgb.png"

    original_image_data = cv2.imread(image_path)

    original_image_shape = np.shape(original_image_data)

    gray_image = cv2.cvtColor(original_image_data, cv2.COLOR_BGR2GRAY)

    gray_image_shape = np.shape(gray_image)
    gray_image = np.float32(gray_image) / 255.0

    print(gray_image_shape)
    print(original_image_shape)

    # for i in range(0,2):

    reduced_image = []
    
    for row in range(0,gray_image_shape[0], 2):
        reduced_row = []
        first_x = row
        second_x = row + 1
        for col in range(0, gray_image_shape[1], 2):
            first_y = col
            second_y = col + 1
            
            first_px = gray_image[first_x][first_y]
            second_px = gray_image[second_x][first_y]
            third_px = gray_image[first_x][second_y]
            forth_px = gray_image[second_x][second_y]
            average_px = (first_px + second_px + third_px + forth_px)/4

            reduced_row.append(average_px)


        if len(reduced_image) < 1:
            reduced_image = np.array(reduced_row)
        else:
            reduced_image = np.vstack((reduced_image, np.array(reduced_row)))


    reduced_image_shape = np.shape(reduced_image)    
    print(reduced_image_shape)
    print(gray_image.dtype)
    print(reduced_image.dtype)
    # reduced_image = reduced_image.astype(np.uint8)
    cv2.imwrite("reduced_version.jpg", reduced_image*255.0)