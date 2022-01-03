# -*- coding: utf-8 -*-
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
from top_defenitions import clinical_b_values, Net, clinical_basic_path, basic_depth, \
                            vars_depth, clinical_vars_path, clinical_vars_input_path, vars_input_depth
from Least_Square_Solver import Least_Square
from net_training_and_testing import image_snr_calculater, normalized_std_calculater


"""
this function cleans the parameter maps with a mask 
- inputs: images, mask, fill_val.
- output: image after the mask and filling
"""
def activate_mask_on_parameter_maps(image, mask, fill_val):
    masked_image = np.ma.array(image, mask = mask)
    new_image = masked_image.filled(fill_value=fill_val)
    return new_image

"""
this function loads the net, and plots it's parameter maps 
- inputs: the depth, weight's path, net name, input size, and the input for the net
"""
def clinical_data_presentor(depth, path, net_name, input_size, dwi_input):
    net = Net(clinical_b_values, input_size, depth)
    net.load_state_dict(torch.load(path))
        
    net.eval()
    with torch.no_grad():
      _, Dp, Dt, Fp = net(torch.from_numpy(dwi_input.astype(np.float32)))
      
    Dp = Dp.numpy()
    Dt = Dt.numpy()
    Fp = Fp.numpy()
      
    # make sure Dp is the larger value between Dp and Dt
    if np.mean(Dp) < np.mean(Dt):
      Dp, Dt = Dt, Dp
      Fp = 1 - Fp
    
    Dp = activate_mask_on_parameter_maps(Dp,mask_vector,0)
    Dt = activate_mask_on_parameter_maps(Dt,mask_vector,0)
    Fp = activate_mask_on_parameter_maps(Fp,mask_vector,0)
      
    Dp = np.reshape(Dp,(sx,sy)) 
    Dt = np.reshape(Dt,(sx,sy)) 
    Fp = np.reshape(Fp,(sx,sy))    
    
    Dp = np.clip(Dp, 0, 0.4)
    Dt = np.clip(Dt, 0, 0.02)
    Fp = np.clip(Fp, 0, 1)
    
    print(net_name)
    
    #for ind = 15
    print("The Dp normalized std is:", normalized_std_calculater(Dp_classic_solver,40,43,95,98))
    print("The Dt normalized std is:", normalized_std_calculater(D_classic_solver,82,85,121,124))
    print("The Fp normalized std is:", normalized_std_calculater(F_classic_solver,55,58,66,69))
    
    """
    #for ind = 24
    print("The Dp normalized std is:", normalized_std_calculater(Dp_classic_solver,91,94,121,124))
    print("The Dt normalized std is:", normalized_std_calculater(D_classic_solver,91,94,52,55))
    print("The Fp normalized std is:", normalized_std_calculater(F_classic_solver,52,55,111,114))
    """
   
    print()
    
    plt.figure()
    plt.title('{} Dp'.format(net_name))
    plt.imshow(Dp, cmap="gray")
    
    plt.figure()
    plt.title('{} Dt'.format(net_name))
    plt.imshow(Dt, cmap="gray")
    
    plt.figure()
    plt.title('{} Fp'.format(net_name))
    plt.imshow(Fp, cmap="gray")
    
# end of functions
    
    
# define variables
ind = 15
sx = 156
sy = 192
clinical_b_values_numpy = clinical_b_values.numpy().astype(int)
sb = len(clinical_b_values)
dwi_image = np.zeros((sx,sy,sb))
dwi_image_long = np.zeros((sx*sy,sb))
dwi_image_for_nets = np.zeros((sx*sy,sb))

# loading the clinical data
reader = sitk.ImageFileReader()
reader.SetImageIO("VTKImageIO")
for i, b_value in enumerate(clinical_b_values_numpy):
    reader.SetFileName('clinical_data_MDDW_UPPER_AVERAGED/b{}_averaged.vtk'.format(b_value))
    image = reader.Execute()
    Im = sitk.GetArrayFromImage(image)
    dwi_image[:,:,i] = Im[ind,:,:]

# calculates the image approximated snr for the snr input net - 2 options for each index
# for index 15
clinical_SNR = image_snr_calculater(dwi_image[:,:,0],37,40,57,60)
#clinical_SNR = image_snr_calculater(dwi_image[:,:,0],82,85,121,124)

# for index 24
#clinical_SNR = image_snr_calculater(dwi_image[:,:,0],77,80,90,93)
#clinical_SNR = image_snr_calculater(dwi_image[:,:,0],91,94,52,55)

# using mask to leave only useful data
mask = (dwi_image[:,:,0] < 40)
for i in np.arange(sb):
    masked_image = np.ma.array(dwi_image[:,:,i], mask = mask)
    dwi_image[:,:,i] = masked_image.filled(fill_value=0)

"""
# present dwi image if needed
plt.figure()
plt.title("DWI MRI image with b=150")
plt.imshow(dwi_image[:,:,6], cmap="gray")
"""

# the input for the nets
dwi_image_long = np.reshape(dwi_image, (sx*sy, sb))

# normalize the data for nets
S0_vector = dwi_image_long[:,0]
mask_vector = (S0_vector <= 0)
masked_vector = np.ma.array(S0_vector, mask = mask_vector)
S0_vector = masked_vector.filled(fill_value=1)
for i in np.arange(sb):
    dwi_image_for_nets[:,i] = dwi_image_long[:,i] / S0_vector



# Classic solver
dwi_image_classic_solver = np.zeros((sb,sx,sy)) #calssic solver needs shape (b_len,size_x,size_y)
for i in np.arange(sb):
    dwi_image_classic_solver[i,:,:] = dwi_image[:,:,i]
S0_Mat = dwi_image_classic_solver[0,:,:]
F_classic_solver, D_classic_solver, Dp_classic_solver = Least_Square(dwi_image_classic_solver, S0_Mat, clinical_b_values, sx, sy)


# for ind = 15
print("Classic solver")
print("The Dp normalized std is:", normalized_std_calculater(Dp_classic_solver,40,43,95,98))
print("The Dt normalized std is:", normalized_std_calculater(D_classic_solver,82,85,121,124))
print("The Fp normalized std is:", normalized_std_calculater(F_classic_solver,55,58,66,69))
print()

"""
# for ind = 24
print("Classic solver")
print("The Dp normalized std is:", normalized_std_calculater(Dp_classic_solver,91,94,121,124))
print("The Dt normalized std is:", normalized_std_calculater(D_classic_solver,91,94,52,55))
print("The Fp normalized std is:", normalized_std_calculater(F_classic_solver,52,55,111,114))
print()
"""

plt.figure()
plt.title("Classic Solver Dp")
plt.imshow(Dp_classic_solver, cmap="gray")

plt.figure()
plt.title("Classic Solver D")
plt.imshow(D_classic_solver, cmap="gray")

plt.figure()
plt.title("Classic Solver Fp")
plt.imshow(F_classic_solver, cmap="gray")


# The Basic net
clinical_data_presentor(basic_depth, clinical_basic_path, 'Basic net', sb, dwi_image_for_nets)

# The SNR net
clinical_data_presentor(vars_depth, clinical_vars_path, 'SNR net', sb, dwi_image_for_nets)

# The SNR with input net
# adding the SNR as another input
dwi_image_long_snr_input = np.zeros((sx*sy, sb+1))
dwi_image_long_snr_input[:,0:sb] = dwi_image_for_nets
dwi_image_long_snr_input[:,sb] = clinical_SNR/100

clinical_data_presentor(vars_input_depth, clinical_vars_input_path, 'SNR with input net', sb+1, dwi_image_long_snr_input)




