import numpy as np
from matplotlib import pyplot as plt
import cv2, math

def get_log_value(M,N, omega):
    result = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            first_term = -(1/(math.pi*math.pow(omega,4)))
            sum_term = math.pow(u,2) + math.pow(v,2)
            second_term = sum_term/(2*math.pow(omega,2))
            third_term = -sum_term/(2*math.pow(omega,2))
            forth_term = math.exp(third_term)
            total_value = first_term*(1-second_term)*forth_term
            result[u,v] = total_value
    return result

def get_lap_of_gaus(u,v, sigma):
    s2 = sigma**2
    sub = -((u**2) + (v**2))/(2*s2)

    return -(-1 / math.pi * (s2**2)) * (1 +sub) * np.exp(sub)

def build_kernel(dimension, sigma, kernel):
    large = dimension // 2

    linSp = np.linspace(-large, large + 1, dimension)

    X, Y = np.meshgrid(linSp, linSp)

    return kernel(u=X, v=Y, sigma=sigma)

def convolution(image, kernel):
    dimKer = kernel.shape[0]
    large = dimKer // 2

    image = np.pad(image, pad_width=((large, large), (large, large)), mode='constant', constant_values=0) \
            .astype(np.float32)

    result = np.zeros(image.shape)

    for h in range(large, image.shape[0] - large):
        for w in range(large, image.shape[1] - large):
            square = image[h - large:h - large + dimKer, w - large:w - large + dimKer]
            result[h, w] = np.sum(np.multiply(square, kernel))

    # Return the convolved image
    return result[large:-large, large:-large]

def transform(image, sigma):
    kernel_dimensions = 2 * int(4 * sigma + 0.5) + 1
    kernel = build_kernel(dimension=kernel_dimensions, sigma=sigma, kernel=get_lap_of_gaus)

    result = convolution(image, kernel)

    return result.astype(np.uint8)

def apply_log_filter(fft, omega):
    M,N = fft.shape

    transform = np.zeros((fft.shape))

    for u in range(M):
        for v in range(N):
            log = generated_H_uv(u, v, omega)
            transform_value = log*fft[u,v]
            transform[u,v] = transform_value

    return transform

def generated_H_uv(u,v, omega):
    first_term = -(1/(math.pi*math.pow(omega,4)))
    sum_term = math.pow(u,2) + math.pow(v,2)
    second_term = sum_term/(2*math.pow(omega,2))
    third_term = -(1/2)*(sum_term/(math.pow(omega,2)))
    forth_term = math.exp(third_term)
    total_value = first_term*(1-second_term)*forth_term
    return total_value

def lap_kernel(M, N, omega):
    result = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            denominator_1 = math.pi*math.pow(omega,4)
            denominator_2 = 2*math.pow(omega,2)
            nominator_1 = 1
            nominator_2 = u**2 + v**2
            r = (-nominator_1/denominator_1)*(1-(nominator_2/denominator_2))*math.exp((-nominator_2/denominator_2))
            result[u,v] = r
    return result

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/natural_scene.png"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (256,256))
    M, N = image.shape
    plt.imshow(image, cmap='gray')
    plt.show()
    log = transform(image, 1)

    plt.imshow(log, cmap='gray')
    plt.show()

    log = transform(image, 4)

    plt.imshow(log, cmap='gray')
    plt.show()

    log = transform(image, 8)

    plt.imshow(log, cmap='gray')
    plt.show()

    log = transform(image, 16)

    plt.imshow(log, cmap='gray')
    plt.show()

    log = transform(image, 32)

    plt.imshow(log, cmap='gray')
    plt.show()