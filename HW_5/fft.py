from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


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

if __name__ == "__main__":

        image_path = "/Users/marynavek/Projects/ImageProcessing/shape_circle.png"
        
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (256,256))

        
        plt.imshow(image, cmap='gray')
        plt.show()
        centeredImge = compute_centered_image(image)
        time_start = timer()
        fft_im = fft(centeredImge)
        time_end = timer()
        time_elapsed = time_end - time_start
        print(f"Total execution time is: {time_elapsed}")
        fftCenteredNormImge = normalize_by_log(fft_im)
        plt.imshow(fftCenteredNormImge, cmap='gray')
        plt.show()

    
#     inverse = inverse_fft(fft)
#     inv_centered = compute_centered_image(inverse)
#     plt.imshow(inv_centered, cmap='gray')
#     plt.show()
    

  