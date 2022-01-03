# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from top_defenitions import b_values, b_len, criterion, basic_depth, \
                                basic_path, vars_depth, vars_path, vars_input_path, \
                                vars_input_depth, SNR_vec, SNR_len
from net_training_and_testing import Test_net_SNR
from Data_simulations import Create_Matrix, Create_Signal, Noise_Adder



# Simulate IVIM parameter maps
# define parameter values in the three regions
S0_region0, S0_region1, S0_region2 = 1500, 1400, 1600
Dp_region0, Dp_region1, Dp_region2 = 0.02, 0.04, 0.06
Dt_region0, Dt_region1, Dt_region2 = 0.0015, 0.0010, 0.0005
Fp_region0, Fp_region1, Fp_region2 = 0.1, 0.2, 0.3
size = 100
lim1 = 40
lim2 = 20
b_values = b_values.numpy()

# Build ground truth tensor
S0_truth = Create_Matrix(size,lim1,lim2,S0_region0,S0_region1,S0_region2)
Dp_truth = Create_Matrix(size,lim1,lim2,Dp_region0,Dp_region1,Dp_region2)
Dt_truth = Create_Matrix(size,lim1,lim2,Dt_region0,Dt_region1,Dt_region2)
Fp_truth = Create_Matrix(size,lim1,lim2,Fp_region0,Fp_region1,Fp_region2)

ground_truth = np.stack((Dp_truth,Dt_truth,Fp_truth,S0_truth))

# Build test data
signal_truth = Create_Signal(Dt_truth, Dp_truth, Fp_truth, b_values, S0_truth)


test_data = np.zeros((size,size,b_len,SNR_len)) #DNN needs shape (size,size,b_len)
noised = np.zeros((b_len, size,size,SNR_len)) #classic solver neeeds shape (b_len, size, size)
E=0
signal_var = np.var(signal_truth)
for i, SNR in enumerate(SNR_vec): # we want to save a 3D tensor for each var (differnt shaped tensor for DNN/Calssic)    
    std = np.sqrt(signal_var)/SNR
    var = std**2
    noised [:,:,:,i] = Noise_Adder(signal_truth, E, var)
    for j in np.arange(b_len):
          test_data[:,:,j,i] = noised[j,:,:,i]

#load the classic solver values
Classic_D = np.load("classic_solver_estimations/Classic_D.npy")
Classic_Dp = np.load("classic_solver_estimations/Classic_Dp.npy")
Classic_F = np.load("classic_solver_estimations/Classic_F.npy")
Classic_loss = np.load("classic_solver_estimations/Classic_loss.npy")

#generate the nets RMSE and loss values
Basic_Dp, Basic_D, Basic_F, Basic_loss = Test_net_SNR(b_len, basic_depth, criterion, \
                                                  SNR_vec, basic_path, test_data, ground_truth)


Var_Dp, Var_D, Var_F, Var_loss = Test_net_SNR(b_len, vars_depth, criterion, \
                                          SNR_vec, vars_path, test_data, ground_truth)
  
Var_Input_Dp, Var_Input_D, Var_Input_F, Var_Input_loss = Test_net_SNR(b_len+1, vars_input_depth, \
                                                                  criterion, SNR_vec, vars_input_path, test_data, ground_truth)


# plots

plt.plot(SNR_vec, Basic_Dp, label= "Basic Barbieri Net")
#plt.plot(SNR_vec, Classic_Dp, label= "Classic LLS Solver")
plt.plot(SNR_vec, Var_Dp, label= "Net with different noise training")
plt.plot(SNR_vec, Var_Input_Dp, label= "Net with different noise training and SNR as input")
plt.title("Dp rmse Vs SNR for 3 nets")
plt.legend()
plt.xlabel("SNR")
plt.ylabel("Dp RMSE")
plt.show()

plt.plot(SNR_vec,Basic_D, label= "Basic Barbieri Net")
#plt.plot(SNR_vec, Classic_D, label= "Classic LLS Solver")
plt.plot(SNR_vec, Var_D, label= "Net with different noise training")
plt.plot(SNR_vec, Var_Input_D, label= "Net with different noise training and SNR as input")
plt.title("D rmse Vs SNR for 3 nets")
plt.legend()
plt.xlabel("SNR")
plt.ylabel("D RMSE")
plt.show()



plt.plot(SNR_vec, Basic_F, label= "Basic Barbieri Net")
#plt.plot(SNR_vec, Classic_F, label= "Classic LLS Solver")
plt.plot(SNR_vec, Var_F, label= "Net with different noise training")
plt.plot(SNR_vec, Var_Input_F, label= "Net with different noise training and SNR as input")
plt.title("Fp rmse Vs SNR for 3 nets")
plt.legend()
plt.xlabel("SNR")
plt.ylabel("Fp RMSE")
plt.ticklabel_format(useOffset=False)
plt.show()

plt.plot(SNR_vec, Basic_loss, label= "Basic Barbieri Net")
#plt.plot(SNR_vec, Classic_loss, label= "Classic LLS Solver")
plt.plot(SNR_vec, Var_loss, label= "Net with different noise training")
plt.plot(SNR_vec, Var_Input_loss, label= "Net with different noise training and SNR as input")
plt.title("Loss Vs SNR for 3 nets")
plt.legend()
plt.xlabel("SNR")
plt.ylabel("Loss (MSE)")
plt.show()
