import math
import numpy as np
from matplotlib import pyplot as plt
import cv2

def compute_single_2d_FFT(image, u, v, N):
        result = 0 + 0j
        for x in range(N):
            for y in range(N):
                result += (image[x, y] * (math.cos((2*math.pi*(u*x + v*y))/N) - 
                                         (1j*math.sin((2*math.pi*(u*x + v*y))/N))))
        return result

def compute_forward_DFT_no_separation(imge):
        N = imge.shape[0]
        final2DDFT = np.zeros([N, N], dtype=np.complex128)
        for u in range(N):
            for v in range(N):
                final2DDFT[u, v] = compute_single_2d_FFT(imge, u, v, N)
        return ((1.0/(N**2))*final2DDFT)


def compute_single_w(num, denom):
        return math.cos((2*math.pi*num)/denom) - (1j*math.sin((2*math.pi*num)/denom))

def compute_centered_image(imge):
        M, N = imge.shape
        newImge = np.zeros([M, N], dtype=int)
        for x in range(M):
            for y in range(N):
                newImge[x, y] = imge[x, y] * ((-1)**(x+y))

        return newImge

def compute_w(val, denom, oneD=True):
        val = int(val)
        if oneD:
            result = np.zeros([val, 1], dtype=np.complex128)
            for i in range(val):
                result[i] = compute_single_w(i, denom)
        else:
            result = np.zeros([val, val], dtype=np.complex128)
            for i in range(val):
                for j in range(val):
                    result[i, j] = compute_single_w((i+j), denom)
        return result

def fft(imge):
        #Compute size of the given image
        N = imge.shape[0]

        #Compute the FFT for the base case (which uses the normal DFT)
        if N == 2:
            return compute_forward_DFT_no_separation(imge)

        #Divide the original image into even and odd
        imgeEE = np.array([[imge[i,j] for i in range(0, N, 2)] for j in range(0, N, 2)]).T
        imgeEO = np.array([[imge[i,j] for i in range(0, N, 2)] for j in range(1, N, 2)]).T
        imgeOE = np.array([[imge[i,j] for i in range(1, N, 2)] for j in range(0, N, 2)]).T
        imgeOO = np.array([[imge[i,j] for i in range(1, N, 2)] for j in range(1, N, 2)]).T

        #Compute FFT for each of the above divided images
        FeeUV = fft(imgeEE)
        FeoUV = fft(imgeEO)
        FoeUV = fft(imgeOE)
        FooUV = fft(imgeOO)

        #Compute also Ws
        Wu = compute_w(N/2, N)
        Wv = Wu.T #Transpose
        Wuv = compute_w(N/2, N, oneD=False)

        #Compute F(u,v) for u,v = 0,1,2,...,N/2  
        imgeFuv = 0.25*(FeeUV + (FeoUV * Wv) + (FoeUV * Wu) + (FooUV * Wuv))

        #Compute F(u, v+M) where M = N/2
        imgeFuMv = 0.25*(FeeUV + (FeoUV * Wv) - (FoeUV * Wu) - (FooUV * Wuv))

        #Compute F(u+M, v) where M = N/2
        imgeFuvM = 0.25*(FeeUV - (FeoUV * Wv) + (FoeUV * Wu) - (FooUV * Wuv))

        #Compute F(u+M, v+M) where M = N/2
        imgeFuMvM = 0.25*(FeeUV - (FeoUV * Wv) - (FoeUV * Wu) + (FooUV * Wuv))

        imgeF1 = np.hstack((imgeFuv, imgeFuvM))
        imgeF2 = np.hstack((imgeFuMv, imgeFuMvM))
        imgeFFT = np.vstack((imgeF1, imgeF2))

        return imgeFFT 

def normalize_by_log(dftImge):
        dftFourierSpect = compute_spectrum(dftImge)
        
        dftNormFourierSpect = (255.0/ math.log10(255)) * np.log10(1 + (255.0/(np.max(dftFourierSpect))*dftFourierSpect))
        
        return dftNormFourierSpect


def compute_spectrum(dftImge):
        N = dftImge.shape[0]
        
        fourierSpect = np.zeros([N, N], dtype=float)
        for i in range(N):
            for j in range(N):
                v = dftImge[i, j]
                fourierSpect[i, j] = math.sqrt((v.real)**2 + (v.imag)**2)
        return fourierSpect

def inverse_fft(imgeFFT):
        N = imgeFFT.shape[0]
        return np.real(np.conjugate(fft(np.conjugate(imgeFFT)*(N**2)))*(N**2))


def buildKernel(dim, sigma, kernel):
    large = dim // 2
    lin_space = np.linspace(-large, large+1, dim)
    X, Y = np.meshgrid(lin_space, lin_space)

    return kernel(x=X, y=Y, sigma=sigma)

def loG(x, y, sigma):
    s2 = np.power(sigma, 2)
    sub1 = -(np.power(x, 2) + np.power(y,2)) / (2 * s2)
    return -(1 / math.pi * np.power(s2, 2)) * (1 + sub1) * np.exp(sub1)

def convolution(image, kernel):
    dimKer = kernel.shape[0]
    large = dimKer // 2

    image = np.pad(image, pad_width=((large, large), (large, large)), mode = 'constant', constant_values=0).astype(np.float32)

    result = np.zeros(image.shape)

    for h in range(large, image.shape[0]-large):
        for w in range(large, image.shape[1]-large):
            square = image[h - large:h - large + dimKer, w - large:w - large+ dimKer]

            result[h,w] = np.sum(np.multiply(square, kernel))

    return result[large:-large, large:-large]

def detector(image, sigma):
    #sigma value definition
    dim = 2*int(4*sigma + 0.5) +1

    kernel = buildKernel(dim=dim, sigma=sigma, kernel=loG)

    result = convolution(image=image.astype(np.uint8), kernel=kernel)

    return result


if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/lady_face.png"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (256,256))

    # centeredImge = compute_centered_image(image)
    # fft_im = fft(centeredImge)
    # fftCenteredNormImge = normalize_by_log(fft_im)
   
    transform = detector(image, 2)


    # fft_im = np.fft.fft2(image)
    # fft_shift = np.fft.fftshift(fft_im)
    # Apply Gaussian Blur
    # blur = cv2.GaussianBlur(image, (3,3),0)

    # # Apply Laplacian operator in some higher datatype
    # laplacian = cv2.Laplacian(blur,cv2.CV_64F)

    # laplacian1 = laplacian/laplacian.max()
    # laplacian = detector(fft_shift.real, 0)

    # # inverse = inverse_fft(laplacian1)
    # # inv_centered = compute_centered_image(inverse)
    # inv_shift = np.fft.ifftshift(laplacian)
    # inv = np.fft.ifft2(inv_shift)


    plt.imshow(transform, cmap='gray')
    plt.show()


    # ddepth = cv2.CV_16S
    # kernel_size = 3
    # window_name = "Laplace Deneme"
    # # [variables]
    # # [load]
    # # src = cv2.imread(cv2.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image
    # # Check if image is loaded fine
    # # if src is None:
    # #     print ('Error opening image')
    # #     print ('Program Arguments: [image_name -- default deneme.png]')
    # #     return -1
    # # [load]
    # # [reduce_noise]
    # # Remove noise by blurring with a Gaussian filter
    # src = cv2.GaussianBlur(src, (3, 3), 10)
    # # [reduce_noise]
    # # [convert_to_gray]
    # # Convert the image to grayscale
    # # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # # [convert_to_gray]
    # # Create Window
    # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    # # [laplacian]
    # # Apply Laplace function
    # dst = cv2.Laplacian(src, ddepth, ksize=kernel_size)
    # # [laplacian]
    # # [convert]
    # # converting back to uint8
    # abs_dst = cv2.convertScaleAbs(dst)
    # # [convert]
    # # [display]
    # plt.imshow(abs_dst, cmap='gray')
    # plt.show()
    