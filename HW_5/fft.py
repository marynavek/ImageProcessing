from PIL import Image
import numpy as np
import math, cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def computeSinglePoint2DFT(imge, u, v, N):
        """
        A private method that computes a single value of the 2DDFT from a given image.

        Parameters
        ----------
        imge : ndarray
            The input image.
        
        u : ndarray
            The index in x-dimension.
            
        v : ndarray
            The index in y-dimension.

        N : int
            Size of the image.
            
        Returns
        -------
        result : complex number
            The computed single value of the DFT.
        """
        result = 0 + 0j
        for x in range(N):
            for y in range(N):
                result += (imge[x, y] * (math.cos((2*math.pi*(u*x + v*y))/N) - 
                                         (1j*math.sin((2*math.pi*(u*x + v*y))/N))))
        return result

def computeForward2DDFTNoSeparability(imge):
        """
        Computes/generates the 2D DFT by computing without separating the kernels.

        Parameters
        ----------
        imge : ndarray
            The input image to be transformed.

        Returns
        -------
        final2DDFT : ndarray
            The transformed image.
        """
 
        # Assuming a square image
        N = imge.shape[0]
        final2DDFT = np.zeros([N, N], dtype=np.complex128)
        for u in range(N):
            for v in range(N):
                #Compute the DFT value for each cells/points in the resulting transformed image.
                final2DDFT[u, v] = computeSinglePoint2DFT(imge, u, v, N)
        return ((1.0/(N**2))*final2DDFT)


def computeSingleW(num, denom):
        """Computes one value of W from the given numerator and denominator values. """
        return math.cos((2*math.pi*num)/denom) - (1j*math.sin((2*math.pi*num)/denom))

def computeCenteredImage(imge):
        """
        Centers a given image.

        Parameters
        ----------
        imge : ndarray
            Input array that stores the image to be centered.

        Returns
        -------
        newImge : int
            The new and centered version of the input image.
        """
        
        #Compute the dimensions of the image
        M, N = imge.shape
        #centeringMatrix = np.zeros([M, N], dtype=int)
        newImge = np.zeros([M, N], dtype=int)
        for x in range(M):
            for y in range(N):
                newImge[x, y] = imge[x, y] * ((-1)**(x+y))

        #newImge = imge * centeringMatrix
        return newImge

def computeW(val, denom, oneD=True):
        """Computes 1D or 2D values of W from the given numerator and denominator values."""
        # print(val)
        val = int(val)
        if oneD:
            result = np.zeros([val, 1], dtype=np.complex128)
            for i in range(val):
                result[i] = computeSingleW(i, denom)
        else:
            result = np.zeros([val, val], dtype=np.complex128)
            for i in range(val):
                for j in range(val):
                    result[i, j] = computeSingleW((i+j), denom)
        return result

def computeFFT(imge):
        """Computes the FFT of a given image.
        """

        #Compute size of the given image
        N = imge.shape[0]

        #Compute the FFT for the base case (which uses the normal DFT)
        if N == 2:
            return computeForward2DDFTNoSeparability(imge)

        #Otherwise compute FFT recursively

        #Divide the original image into even and odd
        imgeEE = np.array([[imge[i,j] for i in range(0, N, 2)] for j in range(0, N, 2)]).T
        imgeEO = np.array([[imge[i,j] for i in range(0, N, 2)] for j in range(1, N, 2)]).T
        imgeOE = np.array([[imge[i,j] for i in range(1, N, 2)] for j in range(0, N, 2)]).T
        imgeOO = np.array([[imge[i,j] for i in range(1, N, 2)] for j in range(1, N, 2)]).T

        #Compute FFT for each of the above divided images
        FeeUV = computeFFT(imgeEE)
        FeoUV = computeFFT(imgeEO)
        FoeUV = computeFFT(imgeOE)
        FooUV = computeFFT(imgeOO)

        #Compute also Ws
        Wu = computeW(N/2, N)
        Wv = Wu.T #Transpose
        Wuv = computeW(N/2, N, oneD=False)

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

def normalize2DDFTByLog(dftImge):
        """
        Computes the log transformation of the transformed DFT image to make the range
        of the fourier values b/n 0 to 255
        
        Parameters
        ----------
        dftImge : ndarray
            The input transformed image.

        Returns
        -------
        dftNormImge : ndarray
            The normalized version of the transformed image.
        """
        
        #Compute the fourier spectrum of the transformed image:
        dftFourierSpect = compute2DDFTFourierSpectrum(dftImge)
        
        #Normalize the fourier spectrum values:
        dftNormFourierSpect = (255.0/ math.log10(255)) * np.log10(1 + (255.0/(np.max(dftFourierSpect))*dftFourierSpect))
        
        return dftNormFourierSpect


def compute2DDFTFourierSpectrum(dftImge):
        """
        Computes the fourier spectrum of the transformed image.

        Parameters
        ----------
        dftImge : ndarray
            The input transformed image.

        Returns
        -------
        fourierSpect : ndarray
            The computed fourier spectrum.
        """
        N = dftImge.shape[0]
        
        fourierSpect = np.zeros([N, N], dtype=float)
        #Calculate the magnitude of each point(complex number) in the DFT image
        for i in range(N):
            for j in range(N):
                v = dftImge[i, j]
                fourierSpect[i, j] = math.sqrt((v.real)**2 + (v.imag)**2)
        return fourierSpect

def fft_2d(img):
        
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

    image_path = "/Users/marynavek/Projects/ImageProcessing/shape_circle.png"
    
    # image = Image.open(image_path).convert("L")
    # img = Image.open(image_path).convert("L")
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (32,32))
    print(image.shape)
    # image = np.zeros((16,16))

    # for M in range(image.shape[0]):
    #     for N in range(image.shape[1]):
    #         print(M)
            # image[M,N] = [M,N]

    centeredImge = computeCenteredImage(image)
    fft = computeFFT(centeredImge)
    fftCenteredNormImge = normalize2DDFTByLog(fft)
    # plt.imshow(compute2DDFTFourierSpectrum(fft), cmap='gray')
    plt.imshow(fftCenteredNormImge, cmap='gray')
    plt.show()
    # get_kernel(64)
    

  