# 1.	Finding the negative of an image: Take an input image f(x,y) and find its negative by using the expression |f(x,y-255| or simply 255 – f(x,y) as an exercise to read and write an image and manipulate its pixels

import cv2
import numpy as np
def grayscale_image(image, image_name):
    height, width = image.shape
    
    new_image = np.empty((image.shape), np.float64)

    for i in range(height):
        for j in range(width):
            
            pixel = 255 - image[i,j]
            new_image[i, j, ...] = pixel

    cv2.imwrite(image_name, new_image)

def rgb_image(image, image_name):
    height, width, channel = image.shape
    
    new_image = np.empty((image.shape), np.float64)

    for i in range(height):
        for j in range(width):
            
            pixel_r = 255 - image[i,j, 0]
            pixel_g = 255 - image[i,j, 1]
            pixel_b = 255 - image[i,j, 2]

            new_image[i, j, 0] = pixel_r
            new_image[i, j, 1] = pixel_g
            new_image[i, j, 2] = pixel_b

    cv2.imwrite(image_name, new_image)

image_p_1 = "/Users/marynavek/Projects/ImageProcessing/HW_1/structure.jpg"
image_p_2 = "/Users/marynavek/Projects/ImageProcessing/HW_1/random.png"
image_p_3 = "/Users/marynavek/Projects/ImageProcessing/HW_1/lady_rgb.png"
image_p_4 = "/Users/marynavek/Projects/ImageProcessing/HW_1/city_rgb.jpg"

image_1 = cv2.imread(image_p_1)
print(image_1.shape)
if len(image_1.shape) < 3:
    
    grayscale_image(image_1, "structure_gray_transform.jpg")
else:
    rgb_image(image_1, "structure_rgb_transform.jpg")
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("structure_gray_version.jpg", gray_image_1)
    grayscale_image(gray_image_1, "structure_gray_transform.jpg")

image_2 = cv2.imread(image_p_2)
print(image_2.shape)
if len(image_2.shape) < 3:
    grayscale_image(image_2, "random_gray_transform.jpg")
else:
    rgb_image(image_2, "random_rgb_transform.jpg")
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("random_gray_version.jpg", gray_image_2)
    grayscale_image(gray_image_2, "random_gray_transform.jpg")

image_3 = cv2.imread(image_p_3)
print(image_3.shape)
if len(image_3.shape) < 3:
    grayscale_image(image_3, "lady_gray_transform.jpg")
else:
    rgb_image(image_3, "lady_rgb_transform.jpg")
    gray_image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("lady_gray_version.jpg", gray_image_3)
    grayscale_image(gray_image_3, "lady_gray_transform.jpg")

image_4 = cv2.imread(image_p_4)
print(image_4.shape)
if len(image_4.shape) < 3:
    grayscale_image(image_4, "city_gray_transform.jpg")
else:
    rgb_image(image_4, "city_rgb_transform.jpg")
    gray_image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("city_gray_version.jpg", gray_image_4)
    grayscale_image(gray_image_4, "city_gray_transform.jpg")



# path_1 = "/Users/marynavek/Projects/ImageProcessing/HW_1/image_1.jpg"
# image = cv2.imread(path_1)
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# rgb_image(image, "rgb_transorm.jpg")
# grayscale_image(gray_image, "gray_transorm.jpg")


# cv2.imwrite("gray_image.jpg", gray_image)

# height, width = gray_image.shape
# print(gray_image.shape)
# new_image = np.empty((gray_image.shape), np.float64)

# for i in range(height):
#     for j in range(width):
        
#         pixel = 255 - gray_image[i,j]
#         new_image[i, j, ...] = pixel

# cv2.imwrite("negative_transorm.jpg", new_image)



