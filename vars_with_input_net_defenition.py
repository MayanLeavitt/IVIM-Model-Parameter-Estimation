# -*- coding: utf-8 -*-
# import libraries
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as utils
from top_defenitions import num_samples_var, SNR_set_size, b_values, clinical_b_values, b_len, ivim,\
                                Net, criterion, vars_input_path, vars_input_depth, SNR_vec, clinical_vars_input_path
from net_training_and_testing import Train_net

"""
# for working with the clinical data - turn off when working with simulated data
b_len = len(clinical_b_values)
b_values = clinical_b_values
"""

# buiding the training data
X_train = np.zeros((num_samples_var, b_len+1)) #last colum is for SNR
X_train_real = np.zeros((num_samples_var,b_len-1))
X_train_imag = np.zeros((num_samples_var,b_len-1))
for i in range(len(X_train)):
    Dp = np.random.uniform(0.01, 0.1)
    Dt = np.random.uniform(0.0005, 0.002)
    Fp = np.random.uniform(0.1, 0.4)
    X_train[i, 0:b_len] = ivim(b_values, Fp, Dt, Dp)
 
    
# add some noise
# all the samples of the set have noise with the same var 
signal_var = np.var(X_train[:,0:b_len])
for i, SNR in enumerate(SNR_vec):
    std = np.sqrt(signal_var)/SNR
    X_train_real[i*SNR_set_size:(i+1)*SNR_set_size,:] = X_train[i*SNR_set_size:(i+1)*SNR_set_size,1:b_len] + np.random.normal(scale=std, size=(SNR_set_size,b_len-1))
    X_train_imag[i*SNR_set_size:(i+1)*SNR_set_size,:] = np.random.normal(scale=std, size=(SNR_set_size, b_len-1))
    X_train[i*SNR_set_size:(i+1)*SNR_set_size,1:b_len] = np.sqrt(X_train_real[i*SNR_set_size:(i+1)*SNR_set_size,:]**2 + X_train_imag[i*SNR_set_size:(i+1)*SNR_set_size,:]**2)
    X_train[i*SNR_set_size:(i+1)*SNR_set_size,b_len] = SNR/100


#instantiate the neural network
#the input vector includes the noise variance, so the vector is longer by one
Vars_With_Input_Net = Net(b_values, b_len+1, vars_input_depth)

# optimizer
optimizer = optim.Adam(Vars_With_Input_Net.parameters(), lr = 0.001)

# training data
batch_size = 128
num_batches = len(X_train) // batch_size
trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)), #EM:organize data in batches
                                batch_size = batch_size, 
                                shuffle = True,
                                drop_last = True)

#Train Net
patience = 10
Train_net(Vars_With_Input_Net, trainloader, optimizer, criterion, patience, vars_input_path) #for simulated data
#Train_net(Vars_With_Input_Net, trainloader, optimizer, criterion, patience, clinical_vars_input_path) #for clinical data