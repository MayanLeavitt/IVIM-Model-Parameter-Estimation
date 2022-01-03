# -*- coding: utf-8 -*-

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from top_defenitions import ivim


"""
this function buils a linear model of points(b,ln(Si/S0)) and return D,F paramters prediction.
- input Si: vector of size len(b), of signal values (only for b>=200).
- input S0: S0 value.
- input b: vector of b values.
- output F: the prediction of D according to built model.
- output D: the prediction of D according to built model.
"""
def segmented_least_squares(Si, S0, b):
    b_len = len(b)
    x_train = np.ones((b_len,2))
    x_train[:,1] = b
    y_train = np.log(Si/S0)
    model = LinearRegression(fit_intercept=(False))
    model.fit(x_train, y_train)
    c1,c2 = model.coef_
    F = (1 - np.exp(c1))
    D = -c2
    return F, D


"""
this function buils a model of points(b,Si/S0) and return F,D,Dp paramters prediction.
- input Si: vector of size len(b), of signal values.
- input S0: S0 value.
- input b: vector of b values.
- input F: initial value of F from segmented_least_squares.
- input D: initial value of D from segmented_least_squares.
- output p_opt: the prediction of F,D,Dp according to built model.
"""
def full_least_squares(Si, S0, b, F, D):
    Dp = D*10
    p0 = np.array([F, D, Dp])
    p_opt,_ = curve_fit(ivim, b, Si/S0, p0, maxfev=20000)#, bounds=([0.0005, 0.000045, 0.00034], [0.9995, 0.018, 1]))
    return p_opt
#can't use bounds when comparing to nets      

"""
this function incorporates two least square methods into final algorithem.
- input Si: tensor of signal values.
- input S0: matrix of S0 values.
- input b: vector of b values going from small to large.
- output F: the parameter map for F.
- output D: the parameter map for D.
- output Dp: the parameter map for Dp.
"""
def Least_Square(Si, S0, b, size_x, size_y):
    F = np.zeros((size_x,size_y))
    D = np.zeros((size_x,size_y))
    Dp = np.zeros((size_x,size_y))
    count = 0
    for i in b:    
        if (i > 200):
            count+=1
    index = len(b) - count
    for i in range(size_x):
        for j in range(size_y):
            if ((S0[i,j] != 0) and (0 not in Si[:,i,j])):
                F_init, D_init = segmented_least_squares(Si[index:,i,j], S0[i,j], b[index:])
                [F_init, D_init] = [0, 0] if (F_init < 0 or D_init < 0) else [F_init, D_init]
                F[i,j], D[i,j], Dp[i,j] = full_least_squares(Si[:,i,j], S0[i,j], b, F_init, D_init)
            else:
                F[i,j], D[i,j], Dp[i,j] = 0, 0, 0
    return F, D, Dp


"""
this function calculates and plots the absolute difference of the approximated matrices and the original matrices.
- input D_Mat: the matrix with the original D parameter values.
- input Dp_Mat: the matrix with the original Dp parameter values.
- input F_Mat: the matrix with the original F parameter values.
- input D_Map: the matrix with the approximated D parameter values.
- input Dp_Map: the matrix with the approximated Dp parameter values.
- input F_Map: the matrix with the approximated F parameter values.
"""
def Matrix_Difference(D_Mat, Dp_Mat, F_Mat, D_Map, Dp_Map, F_Map):
    D_diff = np.abs(D_Mat - D_Map)
    Dp_diff = np.abs(Dp_Mat - Dp_Map)
    F_diff = np.abs(F_Mat - F_Map)
    
    plt.figure()
    plt.title("D Difference")
    plt.imshow(D_diff, cmap='viridis')
    plt.colorbar()
    
    plt.figure()
    plt.title("Dp Difference")
    plt.imshow(Dp_diff, cmap="gray")
    
    plt.figure()
    plt.title("F Difference")
    plt.imshow(F_diff, cmap="gray")
    





