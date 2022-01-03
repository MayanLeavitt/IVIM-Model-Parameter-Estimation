# -*- coding: utf-8 -*-
# import libraries
import numpy as np
import torch
import math
from tqdm import tqdm
from top_defenitions import Net, b_values, b_len, clinical_b_values, RMSE_Calculator
from Data_simulations import Create_Signal
import matplotlib.pyplot as plt

"""
# for working with the clinical data - turn off when working with simulated data
b_len = len(clinical_b_values)
"""

"""
This function initiates the DNN training.
- input net: the network to train.
- input patience: the number of bad epochs to indure before ending training.
- input path: the path to save the weights (including .pth).
"""
def Train_net(net, trainloader, optimizer, criterion, patience, path):
    # Best loss
    best = 1e16
    num_bad_epochs = 0
    loss_vector = np.zeros((1,1000))
    Dp_vector = np.zeros((1,1000))
    Dt_vector = np.zeros((1,1000))
    Fp_vector = np.zeros((1,1000))
    
    # Train
    for epoch in range(1000): 
        print("-----------------------------------------------------------------")
        print("Epoch: {}; Bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0
    
        for i, X_batch in enumerate(tqdm(trainloader), 0):
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            X_pred, Dp_pred, Dt_pred, Fp_pred = net(X_batch)
            loss = criterion(X_pred, X_batch[:,0:b_len]) #EM: although the networks output is the parameters, we calculate MSE using the whole signal
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
     #EM: we calculate loss for all the samples in the batch and sum the loss over all batches.
     #EM: after running on all batches in the epoch we use the loss sum to make decisions.
    
        loss_vector[0,epoch] = running_loss 
        Dp_vector[0,epoch] = Dp_pred[0]
        Dt_vector[0,epoch] = Dt_pred[0]
        Fp_vector[0,epoch] = Fp_pred[0]
    
    #EM: saves the prediction of the last batch in the epoch
    
        print("\n Loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("############### Saving good model ###############################")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("Done, best loss: {}".format(best))
                break
    print("Done")
    # Restore best model
    net.load_state_dict(final_model)    
    # saving basic net best model
    torch.save(net.state_dict(), path)
    


"""
This function initiates the network testing. We test the network's performance for different noise 
s.
- inputs, depth, criterion are used to recover the network.
- SNR_vec: the different SNR's we use to test the network.
- path: the path where the weights are stored. 
- test_data: a tensor of size (sx,sy,sb,SNR_len) that contains noised test data,
such that we have a 3D tensor for each var.
- ground_truth: a tensor of size (sx,sy,4) that contains the parameter maps (truth) 
in the following order: Dp,D,F,S0.
"""     
    
def Test_net_SNR(inputs_len, depth, criterion, SNR_vec, path, test_data, ground_truth): 
    # Restore best basic net
    net = Net(b_values, inputs_len, depth)
    net.load_state_dict(torch.load(path))
    
        
    # image size
    sx, sy, sb = 100, 100, len(b_values)


    #EM: calculate the loss and rmse for different vars of the noise  
    loss_vector_per_SNR = np.zeros(len(SNR_vec))
    rmse_D_vector_per_SNR = np.zeros(len(SNR_vec))
    rmse_Dp_vector_per_SNR = np.zeros(len(SNR_vec))
    rmse_F_vector_per_SNR = np.zeros(len(SNR_vec))   
      
    for i, SNR in enumerate(SNR_vec):
      # inference
      # normalize signal
      dwi_image_long = np.zeros((sx*sy, sb+1))
      dwi_image_long[:,0:sb] = np.reshape(test_data[:,:,:,i], (sx*sy, sb))
      S0 = np.expand_dims(dwi_image_long[:,0], axis=-1)
      dwi_image_long = dwi_image_long/S0 #EM: we normalize by S0 beacuse the network works with normalized signals
      dwi_image_long[:,sb] = SNR/100
    
      net.eval()
      with torch.no_grad():
        predicted_signal, Dp, Dt, Fp = net(torch.from_numpy(dwi_image_long[:,0:inputs_len].astype(np.float32)))
    
    
      Dp = Dp.numpy()
      Dt = Dt.numpy()
      Fp = Fp.numpy()
    
    
      # make sure Dp is the larger value between Dp and Dt
      if np.mean(Dp) < np.mean(Dt):
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
        
      """ 
      # present estimation for a certain SNR.  
      if(SNR == 40):
        plt.figure()
        plt.title("Basic Net Dp Estimation")
        plt.imshow(np.reshape(Dp, (sx, sy)), cmap="gray")
        plt.colorbar()
        
        plt.figure()
        plt.title("Basic Net D Estimation")
        plt.imshow(np.reshape(Dt, (sx, sy)), cmap="gray")
        plt.colorbar()
        
        plt.figure()
        plt.title("Basic Net Fp Estimation")
        plt.imshow(np.reshape(Fp, (sx, sy)), cmap="gray")
        plt.colorbar() 
      """
                 
      # calculate rmse for each parameter 
      rmse_Dp_vector_per_SNR[i] = RMSE_Calculator(ground_truth[0],np.reshape(Dp, (sx, sy)))#/np.mean(ground_truth[0])
      rmse_D_vector_per_SNR[i] = RMSE_Calculator(ground_truth[1],np.reshape(Dt, (sx, sy)))#/np.mean(ground_truth[1])
      rmse_F_vector_per_SNR[i] = RMSE_Calculator(ground_truth[2],np.reshape(Fp, (sx, sy)))#/np.mean(ground_truth[2])
    
      # calculate loss
      signal_truth = Create_Signal(ground_truth[1], ground_truth[0], ground_truth[2], b_values.numpy(), ground_truth[3])
      S0_duplicate = np.reshape(np.array([ground_truth[3]]*sb),(sx*sy,sb))
      signal_normalized = np.reshape(signal_truth, (sx*sy, sb))/S0_duplicate
      loss_vector_per_SNR[i] = criterion(predicted_signal, torch.from_numpy(signal_normalized))
     
    return rmse_Dp_vector_per_SNR, rmse_D_vector_per_SNR, rmse_F_vector_per_SNR, loss_vector_per_SNR

"""
This function calculates the approximated SNR of an image.
- input image: the image.
- input limits: a uniform looking square to calculate the snr.
- output: the approximated snr.
"""
def image_snr_calculater(image, x_lim1, x_lim2, y_lim1, y_lim2):
    return np.mean(image[x_lim1:x_lim2,y_lim1:y_lim2])/(math.sqrt(np.var(image[x_lim1:x_lim2,y_lim1:y_lim2])))

"""
This function calculates the approximated normalized std of an image.
- input image: the image.
- input limits: a uniform looking square to calculate the snr.
- output: the approximated normalized std.
"""
def normalized_std_calculater(image, x_lim1, x_lim2, y_lim1, y_lim2):
    return (math.sqrt(np.var(image[x_lim1:x_lim2,y_lim1:y_lim2])))/np.mean(image[x_lim1:x_lim2,y_lim1:y_lim2])

