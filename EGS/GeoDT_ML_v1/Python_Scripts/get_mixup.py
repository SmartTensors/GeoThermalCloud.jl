# WORK-IN-PROGRESS
#
# Pre-computed mixup -- Generate weak supervised labels
# Mixup's lambda is from Beta distribution (https://arxiv.org/pdf/1710.09412.pdf)
# alpha > 0, beta > 0; lam   = np.random.beta(alpha, beta)
# Herein, we assume lam = 0.5, resulting in an average mixup between two samples
#
# Train/Val/Test splits:
#   Val = 10%, Test = 10%
#   Train --> 5%, 10%, 20%, 40%, 60%, and 80%
# Author: Maruti Kumar Mudunuru

import os
import time
import numpy as np
from itertools import combinations

#==============================================================;
#  Function-1: Pre-computed mixup for two sample combinations  ;
#==============================================================;
def mixup_data_twosamples(x1, x2, y1, y2, alpha, beta):

    #-----------------;
    # Generate mixup  ;
    #-----------------;
    #np.random.seed(int.from_bytes(os.urandom(4), byteorder='little')) #Needed for multiprocessing
    #lam   = np.random.beta(alpha, beta) #Sample from beta distribution
    #np.random.dirichlet((10, 5, 3), 20) for multi-dimensional mixup
    lam   = 0.5 #Average between two samples
    x_lam = lam * x1 + (1.0 - lam) * x2 #Mixup inputs vectors
    y_lam = lam * y1 + (1.0 - lam) * y2 #Mixup output vectors

    return x_lam, y_lam, lam

#==========================================================;
#  Function-2: Compute all unique combination from a list  ;
#              (If a len(list) = n, it will be n*(n-1)/2   ;
#               unique combinations for two samples)       ;
#==========================================================;
def get_all_unique_pairs(num_realz, k):
   
    #----------------------------------------------------------------;
    # Generate unique pairs of realizations                          ;
    # (nCk = n! / k! / (n-k)! when 0 <= k <= n or zero when k > n.)  ; 
    #----------------------------------------------------------------;
    realz_list = list(range(0,num_realz)) #List of realization numbers
    comb_list  = np.array([comb for comb in \
                          combinations(realz_list,k)]) #List of all possible unique pairs
    
    return comb_list

#**********************************************************;
#  Mixup for two samples -- Different training data sizes  ;
#     5%  = 522   --> 135,981    samples                   ; 
#     10% = 1043  --> 543,403    samples                   ;
#     20% = 2087  --> 2,176,741  samples                   ;
#     40% = 4175  --> 8,713,225  samples                   ;
#     60% = 6263  --> 19,609,453 samples                   ;
#     80% = 10439 --> 54,481,141 samples                   ;
#     Each input-output has the following dimensions       ;
#       INPUT  = (37,)                                     ; 
#       OUTPUT = (63,)                                     ;
#**********************************************************;
if __name__ == '__main__':

    #=========================;
    #  Start processing time  ;
    #=========================;
    tic = time.perf_counter()
    
    #----------------------------------------------------------;
    #  1. Get raw training data (5%, 10%, 20%, 40%, 60%, 80%)  ;
    #----------------------------------------------------------;
    #path = os.getcwd() #Get current directory path
    path                = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v2/'
    path_raw_data       = path + 'Data/Raw_Data/' #Raw data
    path_raw_data_mixup = path + 'Data/Raw_Data_Mixup/' #Raw data with Mixup
    #
    num_train    = 10439 #Training realz
    #
    x_p          = np.load(path_raw_data + "train_raw_p_" + \
                            str(num_train) + ".npy") #GeoDT params -- 13049 realz; (13049, 63)
    x_hpro       = df_hpro.values #hpro -- 13049 realz; (13049, 37)
    x_pout       = df_pout.values #pout -- 13049 realz; (13049, 37)
    x_dhout      = df_dhout.values #dhout -- 13049 realz; (13049, 37)

    #----------------------;
    #  2. Initializations  ;
    #----------------------;
    np.random.seed(1337) #For reproducability
    #
    realz_list = [522, 1043, 2087, 4175, 6263, 10439]

    #======================;
    #  3. Initializations  ;
    #======================;
    alpha = 0.5 #0.1 to 0.5
    beta  = 0.5 #0.1 to 0.5
    #
    i = 0
    num_realz = realz_list[i]
    k         = 2
    comb_list = get_all_unique_pairs(num_realz,k) #Unique pairs
    print(comb_list)
    print('Num realz and comb = ', num_realz, \
                comb_list.shape, int(0.5*num_realz*(num_realz-1))) #size (0.5*n*(n-1),2)
    print('\n')

    #======================;
    # End processing time  ;
    #======================;
    toc = time.perf_counter()
    print('Time elapsed in seconds = ', toc - tic)