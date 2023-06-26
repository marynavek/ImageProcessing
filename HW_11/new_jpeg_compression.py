import numpy as np
import cv2, math
import collections as cl
from matplotlib import pyplot as plt

# Method for converting an img to 8-bit values
def to_8bits_values(img):
    img_within_range = np.round(((img - np.min(img))*255)/(np.max(img) - np.min(img)))
    return img_within_range

# Method for calculating the entropi of an image
def calculate_entropi(img):

    freq = cl.Counter(img.ravel())
    G = len(freq)
    h = np.zeros(G)
    p = np.zeros(G)

    index = 0
    for val in freq.values():
        h[index] = val
        index += 1

    for i in range(G):
        p[i] = h[i] / len(img.ravel())

    entropi = 0
    for i in range(G):
        if p[i] > 0:
            entropi += p[i] * math.log2(1/p[i])

    return entropi


def img_equals(img1, img2):

    m, n = np.shape(img1)
    u, v = np.shape(img2)

    if m != u or n != v:
        return False

    for x in range(m):
        for y in range(n):

            diff = img1[x,y] - img2[x,y]

            if abs(diff) > 1:
                return False
    return True

def dct_2d(img):
        M, N = np.shape(img)
        dct_result = np.zeros((M, N))
        pi = math.pi

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
                    for y in range(N-1):
                        cos_x = math.cos((2*x+1)*u*pi/(2*M))
                        cos_y = math.cos((2*y+1)*v*pi/(2*N))

                        temp_sum = img[x,y]*cos_x*cos_y

                        sum += temp_sum

                dct_result[u,v] = alpha_u* alpha_v * sum
        return dct_result

def jpeg_compression(image, q, Q):
    block_size = 8
    M,N = image.shape
    image -= 128

    compressed_image = np.zeros((M,N))

    Q = q*Q

    for x in range(0, M, block_size):
        for y in range(0, N, block_size):
            # F = np.zeros((block_size, block_size))

            f = image[x:x + block_size, y:y + block_size]

            dct = dct_2d(f)

            for u in range(dct.shape[0]):
                for v in range(dct.shape[1]):
                    compressed_image[x+u, y+v] = np.round(dct[u,v]/Q[u,v])

    b = 8

    c = calculate_entropi(compressed_image)

    print("New entropi: %f" % c)
    print("Compression rate: %f" % (b/c))
    print("Percentage removed: %f" % (100*(1-c/b)))

    return compressed_image

def inverse_dct_2d(transformed):
        M, N = np.shape(transformed)
        dct_result_inverse = np.zeros((M, N))

        pi = math.pi

        for x in range(M-1):
            for y in range(N-1):
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
                
                        cos_x = math.cos((2*x+1)*u*pi/(2*M))
                        cos_y = math.cos((2*y+1)*v*pi/(2*N))

                        temp_sum = alpha_u * alpha_v*transformed[u,v]*cos_x*cos_y

                        sum += temp_sum 

                dct_result_inverse[x,y] =  sum

        return dct_result_inverse

def jpeg_decompress(image, q, Q):
    block_size = 8
    M,N = image.shape
    image -= 128
    decompressed_image = np.zeros((M, N))

    Q = q * Q

    for u in range(0, M, block_size):
        for v in range(0, N, block_size):
            F = image[u:u + block_size, v:v + block_size]*Q
            inverse_dct = inverse_dct_2d(F)

            for x in range(inverse_dct.shape[0]):
                for y in range(inverse_dct.shape[1]):
                    decompressed_image[u+x, v+y] = inverse_dct[x,y]
    

    decompressed_image += 128

    return decompressed_image

if __name__ == "__main__":

    Q = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]]
        )


    image_path = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpeg"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (128,128))

    plt.imshow(image, cmap='gray')
    plt.show()
    
    q = 1
    compressed_image = jpeg_compression(image, q, Q)

    decompressed_image = jpeg_compression(compressed_image, q, Q)
    plt.imshow(decompressed_image, cmap='gray')
    plt.show()

    cv2.imwrite('processed_img.jpeg', to_8bits_values(decompressed_image))