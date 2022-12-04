from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def walsh_transform(img):
        
        M, N = np.shape(img)
       
        walsh_results = np.zeros((M, N))

        number_of_bits = int(math.log2(N)) 

        time_start = timer()

        for u in range(M-1):
            for v in range(N-1):
                u_bits = decimalToBinary(u)
                v_bits = decimalToBinary(v)
                u_final = fix_binary_to_lenght(u_bits, number_of_bits)
                v_final = fix_binary_to_lenght(v_bits, number_of_bits)
                sum = 0
                for x in range(M-1):
                    for y in range(N-1):
                        f_x_y = img[x,y]
                        x_bits = decimalToBinary(x)
                        y_bits = decimalToBinary(y)
                        x_final = fix_binary_to_lenght(x_bits, number_of_bits)
                        y_final = fix_binary_to_lenght(y_bits, number_of_bits)

                        temp_sum = 0
                        product = 1
                        for i in range(number_of_bits):
                            # print(i)
                            power = int(x_final[i])*int(u_final[number_of_bits-1-i]) + int(y_final[i])*int(v_final[number_of_bits-1-i])
                            # print(f'x is {x_final[i]} and u is {u_final[n-i]}')
                            product = product*((-1)**power)
                        
                        temp_sum = (1/N)*(f_x_y*product)
                        sum += temp_sum
                walsh_results[u,v] = sum

        time_end = timer()
        time_elapsed = time_end - time_start
        print(f"Total execution time is: {time_elapsed}")
        # print(dct_result)
        plt.imshow(walsh_results, cmap='gray')
        plt.show()
        # np.set_printoptions(linewidth=100) # output line width (default is 75)
        # round6 = np.vectorize(lambda m: '{:6.1f}'.format(m))
        # round6(dct_result)
        # plt.imshow(round6(dct_result), cmap='gray')
        # plt.show()

def get_kernel(N):

    walsh = np.zeros((N, N))
    number_of_bits = int(math.log2(N)) 
    count_x = 0
    for x in range(N):
        count_x += 1
        count_y = 0
        for u in range(N):
            count_y += 1
            x_bits = decimalToBinary(x)
            u_bits = decimalToBinary(u)
            x_final = fix_binary_to_lenght(x_bits, number_of_bits)
            u_final = fix_binary_to_lenght(u_bits, number_of_bits)

            product = 1
            for i in range(number_of_bits):
                # print(i)
                power = int(x_final[i])*int(u_final[number_of_bits-1-i])
                # print(f'x is {x_final[i]} and u is {u_final[n-i]}')
                product = product*((-1)**power)

            walsh[x,u] = (1/N)*product
    # print(count_x)
    # print(count_y)
    # plt.imshow(haar, cmap='gray')
    # plt.show()
    return walsh
        
def fix_binary_to_lenght(binary, lenght):
    while len(binary) < lenght:
        binary = '0'+binary
    return binary


def ordered_kernel(walsh_transform):
    skips_number = []
    for row in walsh_transform:
        current_value = row[0]
        skips = 0
        for i in row:
            temp_value = i
            if temp_value != current_value:
                skips += 1
            current_value = temp_value
        skips_number.append(skips)

    ordered_transform = np.zeros((walsh_transform.shape))

    for original_position, ordered_position in enumerate(skips_number):
        # print(original_position)
        # print(ordered_position)
        ordered_transform[ordered_position] = walsh_transform[original_position]
    
    plt.imshow(ordered_transform, cmap='gray')
    plt.show()

def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    return bin(n).replace("0b", "")

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpg"
    
    # image = Image.open(image_path).convert("L")
    # img = Image.open(image_path).convert("L")
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (64,64))

    # print(type(decimalToBinary(7)))
    # print(image.shape)
    # image = np.zeros((16,16))

    # for M in range(image.shape[0]):
    #     for N in range(image.shape[1]):
    #         print(M)
            # image[M,N] = [M,N]

    # dct_2d(image)
    walsh_transform(image)
    # original_kernel = get_kernel(8)
    # ordered_kernel(original_kernel)
    # walsh_transform(image)
    

  