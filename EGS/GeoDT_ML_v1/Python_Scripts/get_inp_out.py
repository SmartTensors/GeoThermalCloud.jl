# GeoDT raw data -- full2w_4078_inputs.csv
#
# INPUTS:
#	117 inputs features (Others/GeoDT_full2w_4078_inputs.txt)
#
# OUTPUTS (High-priority decision point; NPV): 
#	1 column -- NPV
#
# NUMBER OF REALZ -- 4078
#
# AUTHOR: Maruti Kumar Mudunuru

import numpy as np
import pandas as pd
import time

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path             = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
#
df               = pd.read_csv(path + 'Data/full2w_4078.csv') #[4078 rows x 222 columns]
#
df_inp           = df.iloc[:,0:117].copy(deep = True) #[4078 rows x 117 columns]
df_inp.to_csv(path + 'Data/geodt_params.csv') #[4078 rows x 117 columns]
#
df_npv           = df.iloc[:,126].copy(deep = True) #[4078 rows x 1 columns]
df_npv.to_csv(path + 'Data/npv.csv') #[4078 rows x 1 columns]
#
df_hpro          = df.iloc[:,142:182].copy(deep = True) #[4078 rows x 40 columns]
df_hpro.to_csv(path + 'Data/hpro.csv') #[4078 rows x 40 columns]
#
df_hpro          = df.iloc[:,142:182].copy(deep = True) #[4078 rows x 40 columns]
df_hpro.to_csv(path + 'Data/hpro.csv') #[4078 rows x 40 columns]
#
df_pout          = df.iloc[:,182:222].copy(deep = True) #[4078 rows x 40 columns]
df_pout.to_csv(path + 'Data/pout.csv') #[4078 rows x 40 columns]

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)