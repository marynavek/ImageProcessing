from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def dct_2d(img):
        
        M, N = np.shape(img)
       
        dct_result = np.zeros((M, N))

        pi = math.pi
        time_start = timer()

        for u in range(M-1):
            for v in range(N-1):
                if u == 0:
                    alpha_u = math.sqrt(1/M)
                else:
                    alpha_u = math.sqrt(2/M)
                if v == 0:
                    alpha_v = math.sqrt(1/M)
                else:
                    alpha_v = math.sqrt(2/M)

                sum = 0
                for x in range(M-1):
                    # print(x)
                    for y in range(N-1):
                        cos_x = math.cos((2+x+1)*u*pi/(2*M))
                        cos_y = math.cos((2+y+1)*v*pi/(2*N))

                        temp_sum = img[x,y]*cos_x*cos_y

                        sum += temp_sum

                dct_result[u,v] = alpha_u* alpha_v * sum

        time_end = timer()
        time_elapsed = time_end - time_start
        print(f"Total execution time is: {time_elapsed}")
        # print(dct_result)
        plt.imshow(dct_result, cmap='gray')
        plt.show()
        # np.set_printoptions(linewidth=100) # output line width (default is 75)
        # round6 = np.vectorize(lambda m: '{:6.1f}'.format(m))
        # round6(dct_result)
        # plt.imshow(round6(dct_result), cmap='gray')
        # plt.show()

def get_kernel(N):
    pi = math.pi
    dct = np.zeros((N, N))
    for x in range(N):
        dct[0,x] = math.sqrt(2.0/N) / math.sqrt(2.0)
    for u in range(1,N):
        for x in range(N):
            dct[u,x] = math.sqrt(2.0/N) * math.cos((pi/N) * u * (x + 0.5) )
            
    # np.set_printoptions(precision=3)
    # dct
    # new = math.dot(dct, dtc.T)
    plt.imshow(dct, cmap='gray')
    plt.show()
        


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"
    
    # image = Image.open(image_path).convert("L")
    # img = Image.open(image_path).convert("L")
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (64,64))
    print(image.shape)
    # image = np.zeros((16,16))

    # for M in range(image.shape[0]):
    #     for N in range(image.shape[1]):
    #         print(M)
            # image[M,N] = [M,N]

    dct_2d(image)
    # get_kernel(64)
    

  