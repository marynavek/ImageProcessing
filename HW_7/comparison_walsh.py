import numpy as np
import cv2, math
from matplotlib import pyplot as plt
from timeit import default_timer as timer

def mse_between_two_images(original_image, reconstructed_image):
    differences = np.subtract(original_image, reconstructed_image)
    squared_differences = np.square(differences)
    return squared_differences.mean()

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
        return walsh_results


def inverse_walsh_transform(transform):
        
        M, N = np.shape(transform)
       
        invese_walsh_results = np.zeros((M, N))

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
                        f_x_y = transform[x,y]
                        x_bits = decimalToBinary(x)
                        y_bits = decimalToBinary(y)
                        x_final = fix_binary_to_lenght(x_bits, number_of_bits)
                        y_final = fix_binary_to_lenght(y_bits, number_of_bits)

                        temp_sum = 0
                        product = 1
                        for i in range(number_of_bits):
                            power = int(x_final[i])*int(u_final[number_of_bits-1-i]) + int(y_final[i])*int(v_final[number_of_bits-1-i])
                            product = product*((-1)**power)
                        
                        temp_sum = (1/N)*(f_x_y*product)
                        sum += temp_sum
                invese_walsh_results[u,v] = sum

        time_end = timer()
        time_elapsed = time_end - time_start
        print(f"Total execution time is: {time_elapsed}")
        # print(dct_result)
        return invese_walsh_results

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
                power = int(x_final[i])*int(u_final[number_of_bits-1-i])
                product = product*((-1)**power)

            walsh[x,u] = (1/N)*product
    return walsh
        
def fix_binary_to_lenght(binary, lenght):
    while len(binary) < lenght:
        binary = '0'+binary
    return binary

def decimalToBinary(n):
    return bin(n).replace("0b", "")


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/natural_scene.png"

    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (64,64))
    plt.imshow(image, cmap='gray')
    plt.show()

    walsh = walsh_transform(image)
    plt.imshow(walsh, cmap='gray')
    plt.show()

    reverse = inverse_walsh_transform(walsh)
    plt.imshow(reverse, cmap='gray')
    plt.show()

    mse = mse_between_two_images(image,reverse)
    print(f"MSE Walsh: {mse}")

    
