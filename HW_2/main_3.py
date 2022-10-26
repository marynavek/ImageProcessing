from cv2 import reduce
import numpy as np
import cv2, os

def reduce_plane(im_data, first_x, second_x, first_y, second_y, height, width):
    first_px = im_data[first_x][first_y]
    if second_x < height and second_y < width:
        second_px = im_data[second_x][first_y]
        third_px = im_data[first_x][second_y]
        forth_px = im_data[second_x][second_y]
        average_px = (first_px + second_px + third_px + forth_px)/4

    elif second_x < height:
        second_px = im_data[second_x][first_y]
        # third_px = gray_image[first_x][second_y]
        # forth_px = gray_image[second_x][second_y]
        average_px = (first_px + second_px + 0 + 0)/4
    elif second_y < width:
        # second_px = gray_image[second_x][first_y]
        third_px = im_data[first_x][second_y]
        # forth_px = gray_image[second_x][second_y]
        average_px = (first_px + 0 + third_px + 0)/4
    else:
        average_px = (first_px + 0 + 0 + 0)/4

    return average_px

if __name__ == "__main__":
    image_path = "/Users/marynavek/Projects/ImageProcessing/HW_2/structure.jpg"

    original_image_data = cv2.imread(image_path)

    original_image_shape = np.shape(original_image_data)

    # gray_image = cv2.cvtColor(original_image_data, cv2.COLOR_BGR2GRAY)

    height, width, channels = np.shape(original_image_data)

    # print(gray_image_shape)
    # print(original_image_shape)

    # for i in range(0,2):

    reduced_image = []
    reduced_r_plane = []
    reduced_g_plane = []
    reduced_b_plane = []

    blue_plane = original_image_data[:,:,0]

    green_plane = original_image_data[:,:,1]

    red_plane = original_image_data[:,:,2]

    print(original_image_data.shape)
    print(red_plane.shape)

    reduced_blue = []
    for row in range(0, height,2):
        reduced_row = []
        first_x = row
        second_x = row + 1

        for col in range(0, width, 2):
            first_y = col
            second_y = col + 1

            average_pixel = reduce_plane(blue_plane, first_x, second_x, first_y, second_y, height, width) 

            reduced_row.append(average_pixel)

        if len(reduced_blue) < 1:
            reduced_blue = np.array(reduced_row)
        else:
            reduced_blue = np.vstack((reduced_blue, np.array(reduced_row)))  

    reduced_green = []
    for row in range(0, height,2):
        reduced_row = []
        first_x = row
        second_x = row + 1

        for col in range(0, width, 2):
            first_y = col
            second_y = col + 1

            average_pixel = reduce_plane(green_plane, first_x, second_x, first_y, second_y, height, width) 

            reduced_row.append(average_pixel)

        if len(reduced_green) < 1:
            reduced_green = np.array(reduced_row)
        else:
            reduced_green = np.vstack((reduced_green, np.array(reduced_row)))  
    

    reduced_red = []
    for row in range(0, height,2):
        reduced_row = []
        first_x = row
        second_x = row + 1

        for col in range(0, width, 2):
            first_y = col
            second_y = col + 1

            average_pixel = reduce_plane(red_plane, first_x, second_x, first_y, second_y, height, width) 

            reduced_row.append(average_pixel)

        if len(reduced_red) < 1:
            reduced_red = np.array(reduced_row)
        else:
            reduced_red = np.vstack((reduced_red, np.array(reduced_row)))  


    resize_img = np.zeros((reduced_red.shape[0], reduced_red.shape[1], channels),np.uint8)
    resize_img[:,:,0] = reduced_blue
    resize_img[:,:,1] = reduced_green
    resize_img[:,:,2] = reduced_red

    # reduced_image = np.dstack((reduced_r_plane, reduced_g_plane, reduced_b_plane))
    reduced_image_shape = np.shape(resize_img)    
    print(reduced_image_shape)
    cv2.imwrite("reduced_version.jpg", resize_img)