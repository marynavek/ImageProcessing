from matplotlib import pyplot as plt
import numpy as np
import cv2
import cmath
from PIL import Image
from timeit import default_timer as timer

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/natual_im_1.jpg"
    
    image = Image.open(image_path).convert("L")

    image = image.resize((64,64))
    plt.imshow(image, cmap='gray')
    plt.show()
    f = np.asarray(image)
    M, N = np.shape(f) 

    P,Q = M*2-1,N*2-1
    shape = np.shape(f)

    fp = np.zeros((P, Q))

    fp[:shape[0],:shape[1]] = f

    fpc = np.zeros((P, Q))
    for x in range(P):
        for y in range(Q):
            fpc[x,y]=fp[x,y]*np.power(-1,x+y)


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
    
