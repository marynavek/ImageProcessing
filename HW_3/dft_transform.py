from matplotlib import pyplot as plt
import numpy as np
import cv2, argparse
import cmath
from PIL import Image, ImageDraw

from resizing_image import resize_image
from timeit import default_timer as timer

# parser = argparse.ArgumentParser(
#     description='Make predictions with signature network',
#     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

if __name__ == "__main__":
    
    # args = parser.parse_args()
    # image_path = args.image_path
    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"
    
    image = Image.open(image_path).convert("L")

    image = image.resize((128,128))
    #get the pixels of image into array
    f = np.asarray(image)
    M, N = np.shape(f) # (img x, img y)
    #show image
    # plt.imshow(f, cmap='gray')
    # plt.show()

    #padd image with size of P and Q
    P,Q = M*2-1,N*2-1
    shape = np.shape(f)
    #our padded array
    fp = np.zeros((P, Q))
    #import our image into padded array
    fp[:shape[0],:shape[1]] = f
    # plt.imshow(fp, cmap='gray',vmin=0, vmax=255)
    # plt.show()

    fpc = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            fpc[x,y]=fp[x,y]*np.power(-1,x+y)

    # plt.imshow(fpc, cmap='gray',vmin=0, vmax=255)
    # plt.show()


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

    time_start = timer()
    dft2d = DFT2D(fpc)
    time_end = timer()
    time_elapsed = time_end - time_start
    print(f"Total execution time is: {time_elapsed}")
    plt.imshow(dft2d.real, cmap='gray')
    plt.show()
    cv2.imwrite("dft.jpg", dft2d.real)
    
