# -*- coding: utf-8 -*-

import numpy as np


"""
this function builds a symetric matrix for a given paramater.
- input size: the size of the square matrix.
- input lim1: defines the inner square (distance from margins).
- input lim2: defines the middle square(distance from margins), such that lim1>lim2.
- input val1: the inner value.
- input val2: the middle value.
- input val3: the outer value.
- output Mat: the defined matrix.
"""
def Create_Matrix(size,lim1,lim2,val1,val2,val3):
    Mat = np.zeros((size,size))
    for i in range(0,size):
        for j in range(0,size):
            if(i<size-lim1 and j<size-lim1 and i>=lim1 and j>=lim1):
                Mat[i,j] = val1
            elif (i<size-lim2 and j<size-lim2 and i>=lim2 and j>=lim2):
                Mat[i,j] = val2
            else:
                Mat[i,j] = val3
    return Mat

"""
this function builds a matrix of the IVIM signals (Si). Assumes same size matrices.
- input D_Mat: the matrix with the D parameter values.
- input Dp_Mat: the matrix with the Dp parameter values.
- input F_Mat: the matrix with the F parameter values.
- input b: the b value vector.
- input S0_Mat: the matrix of the S0 signal.
- output Mat: the calculated signal tenzor, where depth corresponds with b value.
"""
def Create_Signal(D_Mat, Dp_Mat, F_Mat, b, S0_Mat):
    b_len = len(b)
    Mat_size = len(D_Mat[0])
    Mat = np.zeros((b_len,Mat_size,Mat_size))
    for i in range(0,b_len):
        exp1 = np.exp(-b[i]*Dp_Mat)
        exp2 = np.exp(-b[i]*D_Mat)
        Mat[i] = S0_Mat*(F_Mat*exp1 + (1-F_Mat)*exp2)
    return Mat
    

"""
this function adds gaussian noise to each value of the signal matrix.
- input  Mat: the signal tensor which we add noise to.
- input E: the expectation of the gaussian.
- input Var: the variance of the gaussian.
- output Mat_With_Noise: the signal tensor with the noise.
""" 

def Noise_Adder(Mat, E, Var):
    size = np.ma.size(Mat,1)
    depth = np.ma.size(Mat,0)
    Real_Noise = np.random.normal(E, np.sqrt(Var), (depth, size, size))
    Imag_Noise = np.random.normal(E, np.sqrt(Var), (depth, size, size))
    Mat_Real = Mat + Real_Noise
    Mat_With_Noise = np.sqrt(Mat_Real**2 + Imag_Noise**2)
    return Mat_With_Noise



"""
this function builds a tensor of Si values, including gaussian noise.
- input size: the size of the length and width.
- input lim1: defines the inner square (distance from margins) of the matrices.
- input lim2: defines the middle square(distance from margins) of the matrices, such that lim1>lim2.
- input D: 1X3 vector with the D parameter values.
- input Dp: 1X3 vector with the Dp parameter values.
- input F: 1X3 vector with the F parameter values.
- input b: vector of b values.
- input S0: 1X3 vector with the S0 signal values.
- input E: the expectation of the gaussian noise.
- input Var: the variance of the gaussian noise.
- output: Tensor of Si values, such that b is the depth axis.
"""
def Create_Data(size,lim1,lim2,D,Dp,F,b,S0,E,Var):
    D_Mat = Create_Matrix(size,lim1,lim2,D[0],D[1],D[2])
    Dp_Mat = Create_Matrix(size,lim1,lim2,Dp[0],Dp[1],Dp[2])
    F_Mat = Create_Matrix(size,lim1,lim2,F[0],F[1],F[2])
    S0_Mat = Create_Matrix(size,lim1,lim2,S0[0],S0[1],S0[2])
    Si_Tensor = Create_Signal(D_Mat, Dp_Mat, F_Mat, b, S0_Mat)
    return Noise_Adder(Si_Tensor, E, Var)



