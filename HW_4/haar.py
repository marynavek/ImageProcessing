from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer

def get_kernel(N):
    haar = np.zeros((N, N))
    x_values = []

    for i in range(0, N):
        x = (i)/N
        x_values.append(x)

    min_r = 0
    max_r = int(math.log2(N))
    rm_tuples = []

    for r in range(min_r, max_r):
        min_m = 1
        max_m = 2**r
        for m in range(min_m, max_m+1):
            rm_tuples.append((r,m))
    
    for u in range(N):
        r,m = rm_tuples[u-1]
        min_1_included = (m-1)/(2**r)
        max_1_not_included = (m-0.5)/(2**r)
        min_2_included = (m-0.5)/(2**r)
        max_2_not_included = (m)/(2**r)
        for v in range(N):
            if u == 0:
                haar_value = 1/math.sqrt(N)
            else:
                x = x_values[v]
                if x>= min_1_included and x< max_1_not_included:
                    haar_value = (2**(r/2))/(math.sqrt(N))
                elif x>= min_2_included and x< max_2_not_included:
                    haar_value = -(2**(r/2))/(math.sqrt(N))
                else:
                    haar_value = 0
            haar[u,v] = haar_value*(math.sqrt(N))

    return haar   
        
def fix_binary_to_lenght(binary, lenght):
    while len(binary) < lenght:
        binary = '0'+binary
    return binary


def ordered_kernel(haar_transform):
    skips_number = []
    for row in haar_transform:
        current_value = row[0]
        skips = 0
        for i in row:
            temp_value = i
            if temp_value != current_value:
                skips += 1
            current_value = temp_value
        skips_number.append(skips)
    ordered_transform = np.zeros((haar_transform.shape))
    for original_position, ordered_position in enumerate(skips_number):
        ordered_transform[ordered_position] = haar_transform[original_position]
    
    return ordered_transform

def multiply_matrix(A,B):
  global C
  if  A.shape[1] == B.shape[0]:
    C = np.zeros((A.shape[0],B.shape[1]),dtype = int)
    for row in range(A.shape[0]): 
        for col in range(B.shape[1]):
            for elt in range(len(B)):
              C[row, col] += A[row, elt] * B[elt, col]
    return C
  else:
    return "Sorry, cannot multiply A and B."

def decimalToBinary(n):
    return bin(n).replace("0b", "")

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/shape_circle.png"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (32,32))

    plt.imshow(image, cmap='gray')
    plt.show()
    haar = get_kernel(32)
    plt.imshow(haar, cmap='gray')
    plt.show()
    new_image = multiply_matrix(haar,image)
    plt.imshow(new_image, cmap='gray')
    plt.show()
    reverse = multiply_matrix(np.linalg.inv(haar),new_image)
    plt.imshow(reverse, cmap='gray')
    plt.show()


  