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

def computeInverseFFT(imgeFFT):
        N = imgeFFT.shape[0]
        return np.real(np.conjugate(computeFFT(np.conjugate(imgeFFT)*(N**2)))*(N**2))

if __name__ == "__main__":

    image_path = "/Users/marynavek/Projects/ImageProcessing/shape_circle.png"
    
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (32,32))

    
    plt.imshow(image, cmap='gray')
    plt.show()
    centeredImge = computeCenteredImage(image)
    fft = computeFFT(centeredImge)
    fftCenteredNormImge = normalize2DDFTByLog(fft)
    plt.imshow(fftCenteredNormImge, cmap='gray')
    plt.show()

    
    inverse = computeInverseFFT(fft)
    inv_centered = computeCenteredImage(inverse)
    plt.imshow(inv_centered, cmap='gray')
    plt.show()
    

  