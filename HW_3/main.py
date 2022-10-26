from matplotlib import pyplot as plt
import numpy as np
import cv2, argparse

from resizing_image import rezie_image


# parser = argparse.ArgumentParser(
#     description='Make predictions with signature network',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

if __name__ == "__main__":
    
    # args = parser.parse_args()
    # image_path = args.image_path
    image_path = "/Users/marynavek/Projects/ImageProcessing/HW_3/city_rgb.jpg"
    
    original_image_data = cv2.imread(image_path)

    original_image_shape = np.shape(original_image_data)

    gray_image = cv2.cvtColor(original_image_data, cv2.COLOR_BGR2GRAY)

    # gray_image = np.float32(gray_image) / 255.0
    print(gray_image.shape)
    image_1 = rezie_image(gray_image)
    print(image_1.shape)
    # image_2 = rezie_image(image_1)
    # image_3 = rezie_image(image_2)

    # image = np.float32(image_1) / 255.0

    rows = image_1.shape[0]
    cols = image_1.shape[1]
    t = np.zeros((rows,cols),np.uint8)
    output_img = np.zeros((rows,cols),np.uint8)
    m = np.arange(rows)
    n = np.arange(cols)
    x = m.reshape((rows,1))
    y = n.reshape((cols,1))
    for row in range(0,rows):
        M1 = 1j*np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
        t[row] = np.dot(M1, image_1[row])
    for col in range(0,cols):
        M2 = 1j*np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
        output_img[:,col] = np.dot(M2, t[:,col])


    out_dftma = np.log(np.abs(output_img))
    fft_lena50 = np.fft.fft2(gray_image)
    out_fftma = np.log(np.abs(fft_lena50))

    plt.imshow(fft_lena50)
    plt.show()