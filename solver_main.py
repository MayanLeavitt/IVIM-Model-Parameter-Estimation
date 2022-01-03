# -*- coding: utf-8 -*-

from Data_simulations import Create_Data, Create_Matrix, Create_Signal
from Least_Square_Solver import Least_Square, Matrix_Difference
import matplotlib.pyplot as plt
import numpy as np
from top_defenitions import b_values, criterion, RMSE_Calculator
from compare_solvers import noised, ground_truth, SNR_len, size
from tqdm import tqdm
import torch

"""
this function buils a linear model of points(b,ln(Si/S0)) and return D,F paramters prediction.
- output D: the prediction of D according to built model.
- output F: the prediction of D according to built model.
"""
def simulate_and_solve():
    size = 100
    lim1 = 30
    lim2 = 15
    F = [0.15, 0.25, 0.35]
    D = [0.001, 0.002, 0.003]
    Dp = [0.01, 0.02, 0.03]
    b = [5, 50, 100, 200, 270, 400, 600, 800]
    S0 = [150,200,250]
    E = 0
    Var = 0.5
    Si = Create_Data(size,lim1,lim2,D,Dp,F,b,S0,E,Var)
    print(Si)
    S0_Mat = Create_Matrix(size,lim1,lim2,S0[0],S0[1],S0[2])
    F_map, D_map, Dp_map = Least_Square(Si, S0_Mat, b, size) 
    D_Mat = Create_Matrix(size,lim1,lim2,D[0],D[1],D[2])
    Dp_Mat = Create_Matrix(size,lim1,lim2,Dp[0],Dp[1],Dp[2])
    F_Mat = Create_Matrix(size,lim1,lim2,F[0],F[1],F[2])
   
    plt.figure()
    plt.title("F Prediction")
    plt.imshow(F_map, cmap="gray")
   
    plt.figure()
    plt.title("F Matrix")
    plt.imshow(F_Mat, cmap="gray")
   
    plt.figure()
    plt.title("D Prediction")
    plt.imshow(D_map, cmap="gray")

    plt.figure()
    plt.title("D Matrix")
    plt.imshow(D_Mat, cmap="gray")

    plt.figure()
    plt.title("Dp Prediction")
    plt.imshow(Dp_map, cmap="gray")

    plt.figure()
    plt.title("Dp Matrix")
    plt.imshow(Dp_Mat, cmap="gray")

    plt.figure()
    plt.title("S0_mat")
    plt.imshow(S0_Mat, cmap="gray")
    
    Matrix_Difference(D_Mat, Dp_Mat, F_Mat, D_map, Dp_map, F_map)
    

def snr_sweep(var_len, test_data, ground_truth):
    D_Mat = ground_truth[0]
    Dp_Mat = ground_truth[1]
    F_Mat = ground_truth[2]
    S0_Mat = ground_truth[3]
    D_RMSE = np.zeros(var_len)
    Dp_RMSE = np.zeros(var_len)
    F_RMSE = np.zeros(var_len)
    loss = np.zeros(var_len)  
    for i in (tqdm(np.arange(SNR_len))):   
        Si = test_data[:,:,:,i]
        F_map, D_map, Dp_map = Least_Square(Si, S0_Mat, b_values.numpy(), size, size)
        Si_pred = Create_Signal(D_map, Dp_map, F_map, b_values.numpy(), S0_Mat)
        D_RMSE[i] = RMSE_Calculator(D_Mat, D_map)
        Dp_RMSE[i] = RMSE_Calculator(Dp_Mat, Dp_map)
        F_RMSE[i] = RMSE_Calculator(F_Mat, F_map)
        loss[i] = criterion(torch.from_numpy(Si_pred/S0_Mat),torch.from_numpy(Si/S0_Mat))
    return D_RMSE, Dp_RMSE, F_RMSE, loss
    
#generate and save the values for later comparison
D_RMSE, Dp_RMSE, F_RMSE, loss = snr_sweep(SNR_len, noised, ground_truth)

np.save('classic_solver_estimations/Classic_D', D_RMSE)
np.save('classic_solver_estimations/Classic_Dp', Dp_RMSE)
np.save('classic_solver_estimations/Classic_F', F_RMSE)
np.save('classic_solver_estimations/Classic_loss', loss)

#simulate_and_solve()
#plt.title("Classic Solver Normalized RMSE for Var$\in$[0,2]")
