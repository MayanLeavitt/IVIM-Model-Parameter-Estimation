# -*- coding: utf-8 -*-
# import libraries
import numpy as np
import torch
import torch.nn as nn


# define b values
b_values = torch.FloatTensor([0,10,20,60,150,300,500,1000])
clinical_b_values = torch.FloatTensor([0,50,100,200,400,600,800])
b_len = len(b_values)

# training data
num_samples_basic = 100000
num_samples_var = 1000000

# test according to different noise SNRs
# all the samples of the set have the same noise SNR
SNR_vec = np.arange(20,100,5)
SNR_len = len(SNR_vec)
SNR_set_size = int(num_samples_var /SNR_len)

# Create the neural network
class Net(nn.Module):
    def __init__(self, b_values, inputs_len, depth):
        super(Net, self).__init__()
        b_len = len(b_values)
        self.b_values = b_values
        self.fc_layers = nn.ModuleList()
        self.fc_layers.extend([nn.Linear(inputs_len, b_len), nn.ELU()])
        for i in range(depth-2): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(b_len, b_len), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(b_len, 3)) #EM: define output layer with 3 outputs.

#EM: ELU is an activation function that for x>0 is x, and for x<=0 is exp(x)-1
#EM: the networks input is a vector of len(inputs) and the output is a vector of the 3 parameters. we run for 100000 samples.

    def forward(self, X):
        params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
        Dp = params[:, 0].unsqueeze(1)
        Dt = params[:, 1].unsqueeze(1)
        Fp = params[:, 2].unsqueeze(1)
        
        """
        sig = nn.Sigmoid()
        
        Dp = 0.005 +sig(Dp)*(0.15-0.005)
        Dt = 0.0003 +sig(Dt)*(0.003-0.0003)
        Fp = 0.05 +sig(Fp)*(0.45-0.05)
        """
        
        Dp = torch.abs(Dp)
        Dt = torch.abs(Dt)
        Fp = torch.abs(Fp)
        
        
        
        X = Fp*torch.exp(-self.b_values*Dp) + (1-Fp)*torch.exp(-self.b_values*Dt)
        
        return X, Dp, Dt, Fp

# Loss function
criterion = nn.MSELoss()

# basic net definitions
basic_path = 'weights/simulated/basic_ivim_net_weights.pth'
clinical_basic_path = 'weights/clinical/basic_ivim_net_weights.pth'
basic_depth = 3

# different vars net definitions
vars_path = 'weights/simulated/different_vars_ivim_net_weights.pth'
clinical_vars_path = 'weights/clinical/different_vars_ivim_net_weights.pth'
vars_depth = 3

# different vars with input net definitions
vars_input_path = 'weights/simulated/vars_with_input_ivim_net_weights.pth'
clinical_vars_input_path = 'weights/clinical/vars_with_input_ivim_net_weights.pth'
vars_input_depth = 3

#set random seed
np.random.seed(7)


# define ivim function normalized by S0
def ivim(b, Fp, Dt, Dp):
    return Fp*np.exp(-b*Dp) + (1-Fp)*np.exp(-b*Dt)

"""
this function calculates the normalized RMSE of the an approximated matrix.
- input Data: the matrix with the original values.
- input Predection: the matrix with the approximated values.
- output: Normalized RMSE.
"""
def RMSE_Calculator(Data, Prediction):
    rmse = ((np.sum((Prediction - Data)**2)/(len(Data[0])*len(Data[:])))**(0.5))
    mean = np.mean(Data) #if normalization is needed
    return rmse#/mean
 