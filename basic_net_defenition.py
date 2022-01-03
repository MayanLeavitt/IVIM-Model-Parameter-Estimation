# -*- coding: utf-8 -*-
# import libraries
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as utils
from top_defenitions import num_samples_basic, b_values, clinical_b_values, b_len, \
                                ivim, Net, criterion, basic_path, basic_depth, clinical_basic_path
from net_training_and_testing import Train_net

"""
# for working with the clinical data - turn off when working with simulated data
b_len = len(clinical_b_values)
b_values = clinical_b_values
"""

# buiding the training data
X_train = np.zeros((num_samples_basic, b_len))
for i in range(len(X_train)):
    Dp = np.random.uniform(0.01, 0.1)
    Dt = np.random.uniform(0.0005, 0.002)
    Fp = np.random.uniform(0.1, 0.4)
    X_train[i, :] = ivim(b_values, Fp, Dt, Dp)
#EM: the samples are generated totally random, so are bad inputs for CNN


# add some noise to the training data - not including S0 (b=0)
SNR = 60
signal_var = np.var(X_train)
noise_std = np.sqrt(signal_var)/SNR
print("noise_std", noise_std)
X_train_real = X_train[:,1:b_len] + np.random.normal(scale=noise_std, size=(num_samples_basic, b_len-1))
X_train_imag = np.random.normal(noise_std, size=(num_samples_basic, b_len-1))
X_train[:,1:b_len] = np.sqrt(X_train_real**2 + X_train_imag**2)


#instantiate the neural network
Basic_Net = Net(b_values, b_len, basic_depth)

# optimizer
optimizer = optim.Adam(Basic_Net.parameters(), lr = 0.001)

# training data
batch_size = 128
num_batches = len(X_train) // batch_size
trainloader = utils.DataLoader(torch.from_numpy(X_train.astype(np.float32)), #EM:organize data in batches
                                batch_size = batch_size, 
                                shuffle = True,
                                drop_last = True)


#Train Net
patience = 10
Train_net(Basic_Net, trainloader, optimizer, criterion, patience, basic_path) # for simulated data
#Train_net(Basic_Net, trainloader, optimizer, criterion, patience, clinical_basic_path) #for clinical data


