# Get train/val/test splits for DL analysis
#   4078 realz (full2w_4078_inputs.csv)
#
# Train/Val/Test splits:
#   Val = 10%, Test = 10%
#   Train --> 5%, 10%, 20%, 40%, 60%, and 80%
# AUTHOR: Maruti Kumar Mudunuru

import os
import time
import pandas as pd
import numpy as np

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#======================================================================;
#  Function-1: Create train/val/test splits and indices (80%/10%/10%)  ;
#======================================================================;
def get_splits_indices(num_realz, num_train, num_val, num_test, path_split_indices):

    #--------------------------;
    # Create and shuffle cols  ;
    #--------------------------;
    np.random.seed(1337) #For reproducable results
    realz_list = np.arange(1, num_realz+1) #Range for realizations
    np.random.shuffle(realz_list)
    #
    train_list = realz_list[0:num_train] #Train realizations 80%
    val_list   = realz_list[num_train:num_train+num_val] #Validation realizations 10%
    test_list  = realz_list[num_train+num_val:] #Validation realizations 10%

    #------------------------------------------;
    # save total, train, and val cols indices  ;
    #------------------------------------------;
    np.savetxt(path_split_indices + "Total_Realz_" + str(int(num_realz)) + ".txt", \
               realz_list, fmt = '%d', header = 'All_realizations_IDs')
    np.savetxt(path_split_indices + "Train_Realz_" + str(int(num_train)) + ".txt", \
               train_list, fmt = '%d', header = 'Train_realizations_IDs')
    np.savetxt(path_split_indices + "Val_Realz_" + str(int(num_val)) + ".txt", \
               val_list, fmt = '%d', header = 'Val_realizations_IDs')
    np.savetxt(path_split_indices + "Test_Realz_" + str(int(num_test)) + ".txt", \
               test_list, fmt = '%d', header = 'Test_realizations_IDs')

    return realz_list, train_list, val_list, test_list

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path                = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
path_split_indices  = path + 'Data/Train_Val_Test_Indices/'

#********************************************************************************;
#  2. Create and save train/val/test splits and indices for 80/10/10 percentage  ;
#********************************************************************************;
num_realz = 4078 #No. of realization
num_train = 3278 #int(0.8 * num_realz) #80% training; 3262 realz
num_val   = 400 #int(0.1 * num_realz) #10% validation and remaining 10% testing; 400 realz
num_test  = 400 #num_realz - num_train - num_val #400 realz
#
realz_list, train_list, val_list, test_list = \
get_splits_indices(num_realz, num_train, num_val, num_test, path_split_indices) #80% training

#*****************************************************************************;
#  2. Create and save train splits and indices for 5/10/20/40/60 percentages  ;
#*****************************************************************************;
train_5_list  = train_list[0:204]  #5% training data; int(0.05*4078) = 204
train_10_list = train_list[0:408] #10% training data; int(0.1*4078)  = 408
train_20_list = train_list[0:816] #20% training data; int(0.2*4078)  = 816
train_40_list = train_list[0:1632] #40% training data; int(0.4*4078) = 1632
train_60_list = train_list[0:2446] #60% training data; int(0.6*4078) = 2446
#
np.savetxt(path_split_indices + "Train_Realz_" + str(204) + ".txt", \
            train_5_list, fmt = '%d', header = 'Train_realizations_IDs') #5% training data
np.savetxt(path_split_indices + "Train_Realz_" + str(408) + ".txt", \
            train_10_list, fmt = '%d', header = 'Train_realizations_IDs') #10% training data
np.savetxt(path_split_indices + "Train_Realz_" + str(816) + ".txt", \
            train_20_list, fmt = '%d', header = 'Train_realizations_IDs') #20% training data
np.savetxt(path_split_indices + "Train_Realz_" + str(1632) + ".txt", \
            train_40_list, fmt = '%d', header = 'Train_realizations_IDs') #40% training data
np.savetxt(path_split_indices + "Train_Realz_" + str(2446) + ".txt", \
            train_60_list, fmt = '%d', header = 'Train_realizations_IDs') #60% training data

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)