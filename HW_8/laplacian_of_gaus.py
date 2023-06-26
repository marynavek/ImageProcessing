from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def get_LoG(x,y, sigma):
    nominator = ((y**2)+(x**2)-2*(sigma**2))
    denominator = ((2*math.pi*(sigma**6)))
    exponent = math.exp(-((x**2)+(y**2))/2*(sigma**2))
    LoG = nominator*exponent/denominator
    return LoG

def create_log(sigma, size=3):
    w = math.ceil(float(size)*float(sigma))

    if(w%2 == 0):
        w = w + 1

    l_o_g_mask = []
    
    w_range = int(math.floor(w/2))
    print(w_range)
    print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range(-w_range, w_range+1):
        for j in range(-w_range, w_range+1):
            print("hello")
            l_o_g_mask.append(get_LoG(i,j,sigma))
    print(len(l_o_g_mask))
    print(l_o_g_mask)
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(w,w)
    return l_o_g_mask

def convolve(image, mask):
    width = image.shape[1]
    height = image.shape[0]
    w_range = int(math.floor(mask.shape[0]/2))

    res_image = np.zeros((height, width))

    # Iterate over every pixel that can be covered by the mask
    for i in range(w_range,width-w_range):
        for j in range(w_range,height-w_range):
            # Then convolute with the mask 
            for k in range(-w_range,w_range):
                for h in range(-w_range,w_range):
                    res_image[j, i] += mask[w_range+h,w_range+k]*image[j+h,i+k]
    return res_image

def run_l_o_g(bin_image, sigma_val, size_val=3):
    # Create the l_o_g mask
    print("creating mask")
    l_o_g_mask = create_log(sigma_val, size_val)

    # Smooth the image by convolving with the LoG mask
    print("smoothing")
    l_o_g_image = convolve(bin_image, l_o_g_mask)

    # Display the smoothed imgage
    # blurred = plt.fig.add_subplot(1,4,2)
    # plt.imshow(l_o_g_image, cmap='gray')
    # plt.show()
    return l_o_g_image

    # # Find the zero crossings
    # print("finding zero crossings")
    # z_c_image = z_c_test(l_o_g_image)
    # print(z_c_image)

    # #Display the zero crossings
    # edges = pltfig.add_subplot(1,4,3)
    # edges.imshow(z_c_image, cmap=cm.gray)
    # pylab.show()


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/lady_face.png"
    
    # image = Image.open(image_path).convert("L")
    # img = Image.open(image_path).convert("L")
    image = cv2.imread(image_path, 2)
    image = cv2.resize(image, (64,64))

    # print(type(decimalToBinary(7)))
    # print(image.shape)
    # image = np.zeros((16,16))

    # for M in range(image.shape[0]):
    #     for N in range(image.shape[1]):
    #         print(M)
            # image[M,N] = [M,N]

    # dct_2d(image)
    # haar = get_kernel(64)
    transform = run_l_o_g(image, 1)
    plt.imshow(transform, cmap='gray')
    plt.show()
    
    

  