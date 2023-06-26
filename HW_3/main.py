from matplotlib import pyplot as plt
import numpy as np
import cv2, argparse
import cmath

from resizing_image import resize_image


# parser = argparse.ArgumentParser(
#     description='Make predictions with signature network',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

if __name__ == "__main__":
    
    # args = parser.parse_args()
    # image_path = args.image_path
    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"
    
    image_data = cv2.imread(image_path)

    original_image_shape = np.shape(image_data)

    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

    # gray_image = np.float32(gray_image) / 255.0
    # print(gray_image.shape)
    image_1 = resize_image(gray_image)
    # print(image_1.shape)
    image_1 = resize_image(image_1)
    # # print(image_1.shape)
    image_1 = resize_image(image_1)
    # # # print(image_1.shape)
    image_1 = resize_image(image_1)
    print(image_1.shape)
    



    def DFT2D(padded):
        M,N = np.shape(padded)
        dft2d = np.zeros((M,N),dtype=complex)
        for k in range(M):
            for l in range(N):
                sum_matrix = 0.0
                for m in range(M):
                    for n in range(N):
                        e = cmath.exp(- 2j * np.pi * (float(k * m) / M + float(l * n) / N))
                        sum_matrix +=  padded[m,n] * e
                dft2d[k,l] = sum_matrix
        return dft2d

    image_1 = gray_image.resize((32,32))
    #get the pixels of image into array
    image_1 = np.asarray(image_1)
    M, N = np.shape(f)
    P,Q = M*2-1,N*2-1
    shape = np.shape(image_1)
    #our padded array
    fp = np.zeros((P, Q))
    #import our image into padded array
    fp[:shape[0],:shape[1]] = image_1
    plt.imshow(fp, cmap='gray',vmin=0, vmax=255)
    plt.show()

    fpc = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            fpc[x,y]=fp[x,y]*np.power(-1,x+y)

    plt.imshow(fpc, cmap='gray',vmin=0, vmax=255)
    plt.show()

    f = DFT2D(image_1)
    plt.imshow(f.real, cmap='gray')
    plt.show()

    # image = np.float32(image_1) / 255.0

    # f = np.fft.fft2(image_1)
    # fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))

    # plt.subplot(121),plt.imshow(image_1, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # plt.show()

    # rows = image_1.shape[0]
    # cols = image_1.shape[1]
    # t = np.zeros((rows,cols),np.uint8)
    # output_img = np.zeros((rows,cols),np.uint8)
    # m = np.arange(rows)
    # n = np.arange(cols)
    # x = m.reshape((rows,1))
    # y = n.reshape((cols,1))
    # for row in range(0,rows):
    #     M1 = 1j*np.sin(-2*np.pi*y*n/cols) + np.cos(-2*np.pi*y*n/cols)
    #     t[row] = np.dot(M1, image_1[row])
    # for col in range(0,cols):
    #     M2 = 1j*np.sin(-2*np.pi*x*m/cols) + np.cos(-2*np.pi*x*m/cols)
    #     output_img[:,col] = np.dot(M2, t[:,col])


    # out_dftma = np.log(np.abs(output_img))
    # fft_lena50 = np.fft.fft2(gray_image)
    # out_fftma = np.log(np.abs(fft_lena50))

    # plt.imshow(fft_lena50)
    # plt.show()