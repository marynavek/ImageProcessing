import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy.linalg import inv
import time
import math

# def dct_coeff():
#     T = np.zeros([8,8])
#     for i in range(8):
#         for j in range(8):
#             if i==0:
#                 T[i,j] = 1/np.sqrt(8)
#             elif i>0:
#                 T[i,j] = np.sqrt(2/8)*np.cos((2*j+1)*i*np.pi/16)
#     return T


def dct(M):
    tmp = np.zeros(M.shape)
    mask = np.zeros([8,8])
    for i in range(M.shape[0]//8):
        for j in range(M.shape[1]//8):
            mask = M[8*i:8*i+8,8*j:8*j+8]
            tmp[8*i:8*i+8,8*j:8*j+8] = dct_2d(mask)
            
    return (tmp)


def quantiz_div(a,b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i,j] = np.round(a[i,j]/b[i,j])
    return tmp


def quantiz(D,Q):
    tmp = np.zeros(D.shape)
    mask = np.zeros([8,8])
    for i in range(D.shape[0]//8):
        for j in range(D.shape[1]//8):
            mask = quantiz_div(D[8*i:8*i+8,8*j:8*j+8],Q)
            tmp[8*i:8*i+8,8*j:8*j+8] = mask
    return (tmp)

def decompress_mul(a,b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i,j] = a[i,j]*b[i,j]
    return tmp

def decompress(C,Q):
    R = np.zeros(C.shape) 
    mask = np.zeros([8,8])
    for i in range(C.shape[0]//8):
        for j in range(C.shape[1]//8):
            mask = decompress_mul(C[8*i:8*i+8,8*j:8*j+8],Q)
            R[8*i:8*i+8,8*j:8*j+8] = mask
    
    N = np.zeros(C.shape)
    
    for i in range(R.shape[0]//8):
        for j in range(R.shape[1]//8):
            mask = inverse_dct_2d(R[8*i:8*i+8,8*j:8*j+8])
            N[8*i:8*i+8,8*j:8*j+8] = np.round(mask) + 128*np.ones([8,8])
    
    return N

def quantization_level(n):
    Q50 = np.zeros([8, 8])

    Q50 = np.array([[16, 11, 10, 16, 24, 40, 52, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    Q = np.zeros([8, 8])
    for i in range(8):
        for j in range(8):
            if n > 50:
                Q[i, j] = min(np.round((100-n)/50*Q50[i, j]), 255)
            else:
                Q[i, j] = min(np.round(50/n * Q50[i, j]), 255)
    return Q

def Compress_img(file,level):

    I = cv2.imread(file)
    
    B, G, R = cv2.split(I)
    
    H = I.shape[0]  
    W = I.shape[1]  
    
    print("Image size: ",I.shape)

    B = B - 128*np.ones([H,W])
    G = G - 128*np.ones([H,W])
    R = R - 128*np.ones([H,W])
    
    # T = dct_coeff()
    # T_prime = inv(T)
    Q = quantization_level(level)
    
    D_R = dct(R)
    D_G = dct(G)
    D_B = dct(B)

    tmp = cv2.merge((D_B, D_G, D_R))

    cv2.imwrite('DCT.jpg',tmp)
    
    C_R = quantiz(D_R,Q)
    C_R[C_R==0] = 0
    C_G = quantiz(D_G,Q)
    C_G[C_G==0] = 0
    C_B = quantiz(D_B,Q)
    C_B[C_B==0] = 0
    
    tmp = cv2.merge((C_B,C_G,C_R))

    cv2.imwrite('After_Quantiz.jpg',tmp)
    return C_B,C_G,C_R,Q
    


def Decompress_img(C_B,C_G,C_R,Q):
    N_R = decompress(C_R,Q)
    N_G = decompress(C_G,Q)
    N_B = decompress(C_B,Q)

    N_I = cv2.merge((N_B, N_G, N_R))
    cv2.imwrite('Decompressed.jpg',N_I)


def Evaluate(file):
    
    I = cv2.imread(file)
    
    I1 = cv2.imread("Decompressed.jpg")
    
    m,n,k = I1.shape
    
    rms = np.sqrt(np.sum(np.square(I1-I)))/(m*n)
    
    snr = np.sum(np.square(I1))/np.sum(np.square(I1-I))

    return rms, snr

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

file = "/Users/marynavek/Projects/ImageProcessing/synthetic_im_3.jpeg"
level = 3
print("Filename: ",file)
print("Level of compression: ",level)

print("Compressing....")
start = time.time()
C_B,C_G,C_R,Q = Compress_img(file,level)
time_comp = time.time()
print("Compression Time: ",np.round(time_comp - start,1)," sec")

print("Decompressing...")
Decompress_img(C_B,C_G,C_R,Q)
time_decomp = time.time()

print("Decompression Time: ",np.round(time_decomp - time_comp,1)," sec")

end = time.time()
print("Total: ",np.round(end - start,1)," sec")
rms, snr = Evaluate(file)
print("RMS: ",np.round(rms,4))
print("SNR: ",np.round(snr,4))