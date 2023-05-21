# Pre-process the GeoDT params data and NPV for DL analysis
#   INPUTS: 
#	   GeoDT params --> 117
#   OUTPUTS: 
#      NPV          --> 1
#   
# Pre-processing methods:
#	7 different methods and their impact on accuracy
#		1. StandardScaler
#		2. MinMaxScaler
#		3. MaxAbsScaler
#		4. RobustScaler
#		5. PowerTransformer (Yeo-Johnson) 
#		6. QuantileTransformer (uniform output)
#		7. QuantileTransformer (Gaussian output)
# Train/Val/Test splits:
#   Val = 10%, Test = 10%
#   Train --> 5%, 10%, 20%, 40%, 60%, and 80%
#
# Neglect list
#	TRAIN --> [57, 991, 1091, 1566, 2026, 2128, 2253, 2295, 2299, 2494, 2703, 2792, 3048, 3061]
#	VAL   --> [349]
#	TEST  --> [98, 201, 261]
#
# AUTHOR: Maruti Kumar Mudunuru

import os
import copy
import pickle
import time
import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
import sklearn #'1.0.2'
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

np.set_printoptions(precision=2)
print("sklearn version = ", sklearn.__version__)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#====================================================;
#  Function-1: Plot histogram of pre-processed data  ;
#====================================================;
def plot_histplot(data_list, num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos):

    #----------------------------------;
    #  Pre-processed data (histogram)  ;
    #----------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax.set_xlabel(str_x_label, fontsize = 24, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 24, fontweight = 'bold')
    #plt.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
    #ax.set_xlim([0, 730])
    #ax.set_ylim([0, 80])
    ax.hist(data_list, bins = num_bins, label = label_name, \
    		edgecolor = 'k', alpha = 0.5, color = 'b', density = True)
    #tick_spacing = 100
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.legend(loc = loc_pos)
    fig.tight_layout()
    fig.savefig(fig_name + '.pdf')
    fig.savefig(fig_name + '.png')
    plt.close(fig)

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
np.random.seed(1340) #For reproducable results
#
path           = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
path_geodt     = path + 'Data/' #GeoDT data
path_ind       = path + 'Data/Train_Val_Test_Indices/' #Train/Val/Test indices
path_pp_models = path + 'Data/PreProcess_Models/' #Pre-processing models for standardization
path_pp_data   = path + 'Data/PreProcessed_Data/' #Pre-processed data
path_raw_data  = path + 'Data/Raw_Data/' #Raw data

#**************************************************************;
#  2a. GeoDT data for pre-processing                           ;
#      (make sure if some GeoDT parameters should be non-neg)  ;
#**************************************************************;
df_p         = pd.read_csv(path_geodt + 'geodt_params.csv', index_col = 0) #[4078 rows x 117 columns]
df_hpro      = pd.read_csv(path_geodt + 'hpro.csv', index_col = 0) #[4078 rows x 40 columns]
df_pout      = pd.read_csv(path_geodt + 'pout.csv', index_col = 0) #[4078 rows x 40 columns]
df_npv       = pd.read_csv(path_geodt + 'npv.csv', index_col = 0) #[4078 rows x 1 columns]
#
geodt_p_list = df_p.columns.to_list() #GeoDT params list -- A total of 117; length = 117
t_list       = df_hpro.columns.to_list() #time-step list -- A total of 40; length = 40
#
x_p          = df_p.values #Matrix of values for GeoDT params -- 4078 realz; (4078, 117)
x_hpro       = df_hpro.values #Matrix of values of hpro -- 4078 realz; (4078, 40)
x_pout       = df_pout.values #Matrix of values of pout -- 4078 realz; (4078, 40)
x_npv        = df_npv.values #Matrix of values of npv -- 4078 realz; (4078,)
#
print(np.argwhere(np.isnan(x_hpro)))
print(np.argwhere(np.isnan(x_pout)))
print(np.argwhere(np.isnan(x_npv)))

#************************************************;
#  3. Develop train/val/test splits              ;
#     Val = 10%, Test = 10%					     ;
#     Train --> 5%, 10%, 20%, 40%, 60%, and 80%  ;
#************************************************;
num_realz  = 4078 #No. of realization (total realz data)
num_train  = 3278 #Training realz
num_val    = 400 #Validation realz
num_test   = 400 #Testing realz
#
train_index_list = np.genfromtxt(path_ind + "Train_Realz_" + str(num_train) + ".txt", \
									dtype = int, skip_header = 1) #(3278,)  
val_index_list   = np.genfromtxt(path_ind + "Val_Realz_" + str(num_val) + ".txt", \
									dtype = int, skip_header = 1) #(400,)
test_index_list  = np.genfromtxt(path_ind + "Test_Realz_" + str(num_test) + ".txt", \
									dtype = int, skip_header = 1) #(400,)
#
train_raw_hpro  = x_hpro[train_index_list-1,:] #Raw training data (3278, 40)
val_raw_hpro    = x_hpro[val_index_list-1,:] #Raw val data (400, 40)
test_raw_hpro   = x_hpro[test_index_list-1,:] #Raw test data (400, 40)
#
train_raw_pout  = x_pout[train_index_list-1,:] #Raw training data (3278, 40)
val_raw_pout    = x_pout[val_index_list-1,:] #Raw val data (400, 40)
test_raw_pout   = x_pout[test_index_list-1,:] #Raw test data (400, 40)
#
train_raw_npv   = x_npv[train_index_list-1,:] #Raw training data (3278, 1)
val_raw_npv     = x_npv[val_index_list-1,:] #Raw val data (400, 1)
test_raw_npv    = x_npv[test_index_list-1,:] #Raw test data (400, 1)
#
train_raw_p     = x_p[train_index_list-1,:] #Raw training data (3278, 117)
val_raw_p       = x_p[val_index_list-1,:] #Raw val data (400, 117)
test_raw_p      = x_p[test_index_list-1,:] #Raw test data (400, 117)
#
np.save(path_raw_data + "train_raw_hpro_" + str(num_train) + ".npy", \
		train_raw_hpro) #Save train ground truth (3278, 40) in *.npy file
np.save(path_raw_data + "val_raw_hpro_" + str(num_val) + ".npy", \
		val_raw_hpro) #Save val ground truth (400, 40) in *.npy file
np.save(path_raw_data + "test_raw_hpro_"  + str(num_test) + ".npy", \
		test_raw_hpro) #Save test ground truth (400, 40) in *.npy file
#
np.save(path_raw_data + "train_raw_pout_" + str(num_train) + ".npy", \
		train_raw_pout) #Save train ground truth (3278, 40) in *.npy file
np.save(path_raw_data + "val_raw_pout_" + str(num_val) + ".npy", \
		val_raw_pout) #Save val ground truth (400, 40) in *.npy file
np.save(path_raw_data + "test_raw_pout_"  + str(num_test) + ".npy", \
		test_raw_pout) #Save test ground truth (400, 40) in *.npy file
#
np.save(path_raw_data + "train_raw_npv_" + str(num_train) + ".npy", \
		train_raw_npv) #Save train ground truth (3278, 1) in *.npy file
np.save(path_raw_data + "val_raw_npv_" + str(num_val) + ".npy", \
		val_raw_npv) #Save val ground truth (400, 1) in *.npy file
np.save(path_raw_data + "test_raw_npv_"  + str(num_test) + ".npy", \
		test_raw_npv) #Save test ground truth (400, 1) in *.npy file
#
np.save(path_raw_data + "train_raw_p_" + str(num_train) + ".npy", \
		train_raw_p) #Save train ground truth (3278, 117) in *.npy file
np.save(path_raw_data + "val_raw_p_" + str(num_val) + ".npy", \
		val_raw_p) #Save val ground truth (400, 117) in *.npy file
np.save(path_raw_data + "test_raw_p_"  + str(num_test) + ".npy", \
		test_raw_p) #Save test ground truth (400, 117) in *.npy file

#*******************************************;
#  4. Pre-processing using Standard Scalar  ;
#*******************************************;
p_ss        = StandardScaler() #GeoDT-params standard-scalar
hpro_ss     = StandardScaler() #hpro standard-scalar
pout_ss     = StandardScaler() #pout standard-scalar
npv_ss      = StandardScaler() #npv standard-scalar
#
p_ss.fit(train_raw_p) #Fit standard-scalar for GeoDT-params -- 3278 realz (training)
hpro_ss.fit(train_raw_hpro) #Fit standard-scalar for hpro -- 3278 realz (training)
pout_ss.fit(train_raw_pout) #Fit standard-scalar for pout -- 3278 realz (training)
npv_ss.fit(train_raw_npv) #Fit standard-scalar for npv -- 3278 realz (training)
#
p_name      = path_pp_models + "p_ss_" + str(num_train) + ".sav"
hpro_name   = path_pp_models + "hpro_ss_" + str(num_train) + ".sav"
pout_name   = path_pp_models + "pout_ss_" + str(num_train) + ".sav"
npv_name    = path_pp_models + "npv_ss_" + str(num_train) + ".sav"
pickle.dump(p_ss, open(p_name, 'wb')) #Save the fitted standard-scalar (SS) GeoDT-params model
pickle.dump(hpro_ss, open(hpro_name, 'wb')) #Save the fitted standard-scalar (SS) hpro
pickle.dump(pout_ss, open(pout_name, 'wb')) #Save the fitted standard-scalar (SS) pout
pickle.dump(npv_ss, open(npv_name, 'wb')) #Save the fitted standard-scalar (SS) npv
#
pp_ss          = pickle.load(open(p_name, 'rb')) #Load already created GeoDT-params standard-scalar model
hpro_ss        = pickle.load(open(hpro_name, 'rb')) #Load already created hpro standard-scalar model
pout_ss        = pickle.load(open(pout_name, 'rb')) #Load already created pout standard-scalar model
npv_ss         = pickle.load(open(npv_name, 'rb')) #Load already created npv standard-scalar model
#
train_ss_p     = pp_ss.transform(train_raw_p) #Transform GeoDT-params (3278, 117)
val_ss_p       = pp_ss.transform(val_raw_p) #Transform GeoDT-params (400, 117)
test_ss_p      = pp_ss.transform(test_raw_p) #Transform GeoDT-params (400, 117)
#
train_ss_hpro  = hpro_ss.transform(train_raw_hpro) #Transform train hpro (3278, 40)
val_ss_hpro    = hpro_ss.transform(val_raw_hpro) #Transform val hpro (400, 40)
test_ss_hpro   = hpro_ss.transform(test_raw_hpro) #Transform test hpro (400, 40)
#
train_ss_pout  = pout_ss.transform(train_raw_pout) #Transform train pout (3278, 40)
val_ss_pout    = pout_ss.transform(val_raw_pout) #Transform val pout (400, 40)
test_ss_pout   = pout_ss.transform(test_raw_pout) #Transform test pout (400, 40)
#
train_ss_npv = npv_ss.transform(train_raw_npv) #Transform train npv (3278, 1)
val_ss_npv   = npv_ss.transform(val_raw_npv) #Transform val npv (400, 1)
test_ss_npv  = npv_ss.transform(test_raw_npv) #Transform test npv (400, 1)
#
np.save(path_pp_data + "train_ss_p_" + str(num_train) + ".npy", \
		train_ss_p) #Save train ground truth (3278, 117) in *.npy file
np.save(path_pp_data + "val_ss_p_" + str(num_val) + ".npy", \
		val_ss_p) #Save val ground truth (400, 117) in *.npy file
np.save(path_pp_data + "test_ss_p_"  + str(num_test) + ".npy", \
		test_ss_p) #Save test ground truth (400, 117) in *.npy file
#
np.save(path_pp_data + "train_ss_hpro_" + str(num_train) + ".npy", \
		train_ss_hpro) #Save train ground truth (3278, 40) in *.npy file
np.save(path_pp_data + "val_ss_hpro_" + str(num_val) + ".npy", \
		val_ss_hpro) #Save val ground truth (400, 40) in *.npy file
np.save(path_pp_data + "test_ss_hpro_"  + str(num_test) + ".npy", \
		test_ss_hpro) #Save test ground truth (400, 40) in *.npy file
#
np.save(path_pp_data + "train_ss_pout_" + str(num_train) + ".npy", \
		train_ss_pout) #Save train ground truth (3278, 40) in *.npy file
np.save(path_pp_data + "val_ss_pout_" + str(num_val) + ".npy", \
		val_ss_pout) #Save val ground truth (400, 40) in *.npy file
np.save(path_pp_data + "test_ss_pout_"  + str(num_test) + ".npy", \
		test_ss_pout) #Save test ground truth (400, 40) in *.npy file
#
np.save(path_pp_data + "train_ss_npv_" + str(num_train) + ".npy", \
		train_ss_npv) #Save train ground truth (3278, 1) in *.npy file
np.save(path_pp_data + "val_ss_npv_" + str(num_val) + ".npy", \
		val_ss_npv) #Save val ground truth (400, 1) in *.npy file
np.save(path_pp_data + "test_ss_npv_"  + str(num_test) + ".npy", \
		test_ss_npv) #Save test ground truth (400, 1) in *.npy file
#
counter = 0
train_neglect_list = []
#
for i in train_ss_npv: #Neglect train-counter = 0 to 13
	if i <= -1:
		print('Train = ', counter, i)
		train_neglect_list.append(counter)
	counter = counter + 1

counter = 0
val_neglect_list = []
#
for i in val_ss_npv: #Neglect val-counter = 0
	if i <= -1:
		print('Val = ', counter, i)
		val_neglect_list.append(counter)
	counter = counter + 1

counter = 0
test_neglect_list = []
#
for i in test_ss_npv: #Neglect test-counter = 0 to 2
	if i <= -1:
		print('Test = ', counter, i)
		test_neglect_list.append(counter)
	counter = counter + 1

train_nn_list      = np.array(list(set(range(0,num_train)) - set(train_neglect_list)), dtype = int) #(3264,)
val_nn_list        = np.array(list(set(range(0,num_val)) - set(val_neglect_list)), dtype = int) #(399,)
test_nn_list       = np.array(list(set(range(0,num_test)) - set(test_neglect_list)), dtype = int) #(397,)
#
str_x_label = 'Training hpro values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_hpro_' + str(num_train)
plot_histplot(train_ss_hpro.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation hpro values' 
fig_name    = path + 'Plots/Pre_Process/val_ss_hpro_' + str(num_val)
plot_histplot(val_ss_hpro.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing hpro values' 
fig_name    = path + 'Plots/Pre_Process/test_ss_hpro_' + str(num_test)
plot_histplot(test_ss_hpro.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Training pout values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_pout_' + str(num_train)
plot_histplot(train_ss_pout.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation pout values' 
fig_name    = path + 'Plots/Pre_Process/val_ss_pout_' + str(num_val)
plot_histplot(val_ss_pout.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing pout values' 
fig_name    = path + 'Plots/Pre_Process/test_ss_pout_' + str(num_test)
plot_histplot(test_ss_pout.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Training npv values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_npv_' + str(num_train)
plot_histplot(train_ss_npv[train_nn_list,:].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation npv values' 
fig_name    = path + 'Plots/Pre_Process/val_ss_npv_' + str(num_val)
plot_histplot(val_ss_npv[val_nn_list,:].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing npv values' 
fig_name    = path + 'Plots/Pre_Process/test_ss_npv_' + str(num_test)
plot_histplot(test_ss_npv[test_nn_list,:].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 's3-MinStress (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_s3_' + str(num_train)
plot_histplot(train_ss_p[:,96].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 's3-MinStress (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ss_s3_' + str(num_val)
plot_histplot(val_ss_p[:,96].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 's3-MinStress (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ss_s3_' + str(num_test)
plot_histplot(test_ss_p[:,96].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All GeoDT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Standard Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ss_GeoDT_' + str(num_train)
plot_histplot(train_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All GeoDT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ss_GeoDT_' + str(num_val)
plot_histplot(val_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All GeoDT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ss_GeoDT_' + str(num_test)
plot_histplot(test_ss_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)




"""
#****************************************;
#  5. Pre-processing using MinMaxScaler  ;
#****************************************;
p_mms       = MinMaxScaler() #SWAT-params min-max-scalar
q_mms       = MinMaxScaler() #SWAT-discharge min-max-scalar
#
p_mms.fit(train_raw_p) #Fit min-max-scalar for SWAT-params -- 800 realz (training)
q_mms.fit(train_raw_q) #Fit min-max-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_mms_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_mms_" + str(num_train) + ".sav"
pickle.dump(p_mms, open(p_name, 'wb')) #Save the fitted min-max-scalar SWAT-params model
pickle.dump(q_mms, open(q_name, 'wb')) #Save the fitted min-max-scalar SWAT-discharge model
#
pp_mms       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params min-max-scalar model
qq_mms       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge min-max-scalar model
#
train_mms_p  = pp_mms.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_mms_p    = pp_mms.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_mms_p   = pp_mms.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_mms_q  = qq_mms.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_mms_q    = qq_mms.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_mms_q   = qq_mms.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_mms_q_" + str(num_train) + ".npy", \
		train_mms_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mms_q_" + str(num_val) + ".npy", \
		val_mms_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mms_q_"  + str(num_test) + ".npy", \
		test_mms_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_mms_p_" + str(num_train) + ".npy", \
		train_mms_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mms_p_" + str(num_val) + ".npy", \
		val_mms_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mms_p_"  + str(num_test) + ".npy", \
		test_mms_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_q_' + str(num_train)
plot_histplot(train_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_mms_q_' + str(num_val)
plot_histplot(val_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_mms_q_' + str(num_test)
plot_histplot(test_mms_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_SFTMP_' + str(num_train)
plot_histplot(train_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mms_SFTMP_' + str(num_val)
plot_histplot(val_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mms_SFTMP_' + str(num_test)
plot_histplot(test_mms_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mms_SFTMP_' + str(num_train)
plot_histplot(train_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mms_SFTMP_' + str(num_val)
plot_histplot(val_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mms_SFTMP_' + str(num_test)
plot_histplot(test_mms_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MinMax Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_mms_q_' + str(num_train)
plot_histplot(qq_mms.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#****************************************;
#  6. Pre-processing using MaxAbsScaler  ;
#****************************************;
p_mas       = MaxAbsScaler() #SWAT-params max-abs-scalar
q_mas       = MaxAbsScaler() #SWAT-discharge max-abs-scalar
#
p_mas.fit(train_raw_p) #Fit max-abs-scalar for SWAT-params -- 800 realz (training)
q_mas.fit(train_raw_q) #Fit max-abs-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_mas_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_mas_" + str(num_train) + ".sav"
pickle.dump(p_mas, open(p_name, 'wb')) #Save the fitted max-abs-scalar SWAT-params model
pickle.dump(q_mas, open(q_name, 'wb')) #Save the fitted max-abs-scalar SWAT-discharge model
#
pp_mas       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params max-abs-scalar model
qq_mas       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge max-abs-scalar model
#
train_mas_p  = pp_mas.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_mas_p    = pp_mas.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_mas_p   = pp_mas.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_mas_q  = qq_mas.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_mas_q    = qq_mas.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_mas_q   = qq_mas.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_mas_q_" + str(num_train) + ".npy", \
		train_mas_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mas_q_" + str(num_val) + ".npy", \
		val_mas_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mas_q_"  + str(num_test) + ".npy", \
		test_mas_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_mas_p_" + str(num_train) + ".npy", \
		train_mas_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_mas_p_" + str(num_val) + ".npy", \
		val_mas_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_mas_p_"  + str(num_test) + ".npy", \
		test_mas_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_q_' + str(num_train)
plot_histplot(train_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_mas_q_' + str(num_val)
plot_histplot(val_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_mas_q_' + str(num_test)
plot_histplot(test_mas_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_SFTMP_' + str(num_train)
plot_histplot(train_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mas_SFTMP_' + str(num_val)
plot_histplot(val_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mas_SFTMP_' + str(num_test)
plot_histplot(test_mas_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_mas_SFTMP_' + str(num_train)
plot_histplot(train_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_mas_SFTMP_' + str(num_val)
plot_histplot(val_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_mas_SFTMP_' + str(num_test)
plot_histplot(test_mas_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'MaxAbs Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_mas_q_' + str(num_train)
plot_histplot(qq_mas.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#****************************************;
#  7. Pre-processing using RobustScaler  ;
#****************************************;
p_rs        = RobustScaler(quantile_range = (25, 75)) #SWAT-params robust-scalar
q_rs        = RobustScaler(quantile_range = (25, 75)) #SWAT-discharge robust-scalar
#
p_rs.fit(train_raw_p) #Fit robust-scalar for SWAT-params -- 800 realz (training)
q_rs.fit(train_raw_q) #Fit robust-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_rs_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_rs_" + str(num_train) + ".sav"
pickle.dump(p_rs, open(p_name, 'wb')) #Save the fitted robust-scalar SWAT-params model
pickle.dump(q_rs, open(q_name, 'wb')) #Save the fitted robust-scalar SWAT-discharge model
#
pp_rs       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params robust-scalar model
qq_rs       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge robust-scalar model
#
train_rs_p  = pp_rs.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_rs_p    = pp_rs.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_rs_p   = pp_rs.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_rs_q  = qq_rs.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_rs_q    = qq_rs.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_rs_q   = qq_rs.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_rs_q_" + str(num_train) + ".npy", \
		train_rs_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_rs_q_" + str(num_val) + ".npy", \
		val_rs_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_rs_q_"  + str(num_test) + ".npy", \
		test_rs_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_rs_p_" + str(num_train) + ".npy", \
		train_rs_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_rs_p_" + str(num_val) + ".npy", \
		val_rs_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_rs_p_"  + str(num_test) + ".npy", \
		test_rs_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_q_' + str(num_train)
plot_histplot(train_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_rs_q_' + str(num_val)
plot_histplot(val_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_rs_q_' + str(num_test)
plot_histplot(test_rs_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_SFTMP_' + str(num_train)
plot_histplot(train_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_rs_SFTMP_' + str(num_val)
plot_histplot(val_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_rs_SFTMP_' + str(num_test)
plot_histplot(test_rs_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_rs_SFTMP_' + str(num_train)
plot_histplot(train_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_rs_SFTMP_' + str(num_val)
plot_histplot(val_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_rs_SFTMP_' + str(num_test)
plot_histplot(test_rs_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'Robust Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_rs_q_' + str(num_train)
plot_histplot(qq_rs.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#**********************************************************;
#  8. Pre-processing using PowerTransformer (yeo-johnson)  ;
#**********************************************************;
p_ptyj      = PowerTransformer(method = "yeo-johnson") #SWAT-params power-transformer-yeo-johnson-scalar
q_ptyj      = PowerTransformer(method = "yeo-johnson") #SWAT-discharge power-transformer-yeo-johnson-scalar
#
p_ptyj.fit(train_raw_p) #Fit power-transformer-yeo-johnson-scalar for SWAT-params -- 800 realz (training)
q_ptyj.fit(train_raw_q) #Fit power-transformer-yeo-johnson-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_ptyj_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_ptyj_" + str(num_train) + ".sav"
pickle.dump(p_ptyj, open(p_name, 'wb')) #Save the fitted power-transformer-yeo-johnson-scalar SWAT-params model
pickle.dump(q_ptyj, open(q_name, 'wb')) #Save the fitted power-transformer-yeo-johnson-scalar SWAT-discharge model
#
pp_ptyj       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params power-transformer-yeo-johnson-scalar model
qq_ptyj       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge power-transformer-yeo-johnson-scalar model
#
train_ptyj_p  = pp_ptyj.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_ptyj_p    = pp_ptyj.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_ptyj_p   = pp_ptyj.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_ptyj_q  = qq_ptyj.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_ptyj_q    = qq_ptyj.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_ptyj_q   = qq_ptyj.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_ptyj_q_" + str(num_train) + ".npy", \
		train_ptyj_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ptyj_q_" + str(num_val) + ".npy", \
		val_ptyj_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ptyj_q_"  + str(num_test) + ".npy", \
		test_ptyj_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_ptyj_p_" + str(num_train) + ".npy", \
		train_ptyj_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_ptyj_p_" + str(num_val) + ".npy", \
		val_ptyj_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_ptyj_p_"  + str(num_test) + ".npy", \
		test_ptyj_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_q_' + str(num_train)
plot_histplot(train_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_q_' + str(num_val)
plot_histplot(val_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_q_' + str(num_test)
plot_histplot(test_ptyj_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_SFTMP_' + str(num_train)
plot_histplot(train_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_SFTMP_' + str(num_val)
plot_histplot(val_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_SFTMP_' + str(num_test)
plot_histplot(test_ptyj_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_ptyj_SFTMP_' + str(num_train)
plot_histplot(train_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_ptyj_SFTMP_' + str(num_val)
plot_histplot(val_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_ptyj_SFTMP_' + str(num_test)
plot_histplot(test_ptyj_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'PTYJ Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_ptyj_q_' + str(num_train)
plot_histplot(qq_ptyj.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#*********************************************************;
#  9. Pre-processing using QuantileTransformer (uniform)  ;
#*********************************************************;
p_qtu       = QuantileTransformer(n_quantiles = num_train, output_distribution = "uniform") #SWAT-params quantile-transformer-uniform-scalar
q_qtu       = QuantileTransformer(n_quantiles = num_train, output_distribution = "uniform") #SWAT-discharge quantile-transformer-uniform-scalar
#
p_qtu.fit(train_raw_p) #Fit quantile-transformer-uniform-scalar for SWAT-params -- 800 realz (training)
q_qtu.fit(train_raw_q) #Fit quantile-transformer-uniform-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_qtu_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_qtu_" + str(num_train) + ".sav"
pickle.dump(p_qtu, open(p_name, 'wb')) #Save the fitted quantile-transformer-uniform-scalar SWAT-params model
pickle.dump(q_qtu, open(q_name, 'wb')) #Save the fitted quantile-transformer-uniform-scalar SWAT-discharge model
#
pp_qtu       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params quantile-transformer-uniform-scalar model
qq_qtu       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge quantile-transformer-uniform-scalar model
#
train_qtu_p  = pp_qtu.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_qtu_p    = pp_qtu.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_qtu_p   = pp_qtu.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_qtu_q  = qq_qtu.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_qtu_q    = qq_qtu.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_qtu_q   = qq_qtu.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_qtu_q_" + str(num_train) + ".npy", \
		train_qtu_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtu_q_" + str(num_val) + ".npy", \
		val_qtu_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtu_q_"  + str(num_test) + ".npy", \
		test_qtu_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_qtu_p_" + str(num_train) + ".npy", \
		train_qtu_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtu_p_" + str(num_val) + ".npy", \
		val_qtu_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtu_p_"  + str(num_test) + ".npy", \
		test_qtu_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_q_' + str(num_train)
plot_histplot(train_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_qtu_q_' + str(num_val)
plot_histplot(val_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_q_' + str(num_test)
plot_histplot(test_qtu_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_SFTMP_' + str(num_train)
plot_histplot(train_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_qtu_SFTMP_' + str(num_val)
plot_histplot(val_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_SFTMP_' + str(num_test)
plot_histplot(test_qtu_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtu_SFTMP_' + str(num_train)
plot_histplot(train_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_qtu_SFTMP_' + str(num_val)
plot_histplot(val_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtu_SFTMP_' + str(num_test)
plot_histplot(test_qtu_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTU Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_qtu_q_' + str(num_train)
plot_histplot(qq_qtu.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)

#*********************************************************;
#  10. Pre-processing using QuantileTransformer (normal)  ;
#*********************************************************;
p_qtn       = QuantileTransformer(n_quantiles = num_train, output_distribution = "normal") #SWAT-params quantile-transformer-normal-scalar
q_qtn       = QuantileTransformer(n_quantiles = num_train, output_distribution = "normal") #SWAT-discharge quantile-transformer-normal-scalar
#
p_qtn.fit(train_raw_p) #Fit quantile-transformer-normal-scalar for SWAT-params -- 800 realz (training)
q_qtn.fit(train_raw_q) #Fit quantile-transformer-normal-scalar for SWAT-discharge -- 800 realz (training)
#
p_name      = path_pp_models + "p_qtn_" + str(num_train) + ".sav"
q_name      = path_pp_models + "q_qtn_" + str(num_train) + ".sav"
pickle.dump(p_qtn, open(p_name, 'wb')) #Save the fitted quantile-transformer-normal-scalar SWAT-params model
pickle.dump(q_qtn, open(q_name, 'wb')) #Save the fitted quantile-transformer-normal-scalar SWAT-discharge model
#
pp_qtn       = pickle.load(open(p_name, 'rb')) #Load already created SWAT-params quantile-transformer-normal-scalar model
qq_qtn       = pickle.load(open(q_name, 'rb')) #Load already created SWAT-discharge quantile-transformer-normal-scalar model
#
train_qtn_p  = pp_qtn.transform(train_raw_p) #Transform SWAT-params (800, 20)
val_qtn_p    = pp_qtn.transform(val_raw_p) #Transform SWAT-params (100, 20)
test_qtn_p   = pp_qtn.transform(test_raw_p) #Transform SWAT-params (100, 20)
#
train_qtn_q  = qq_qtn.transform(train_raw_q) #Transform SWAT-discharge params (800, 1096)
val_qtn_q    = qq_qtn.transform(val_raw_q) #Transform SWAT-discharge params (100, 1096)
test_qtn_q   = qq_qtn.transform(test_raw_q) #Transform SWAT-discharge params (100, 1096)
#
np.save(path_pp_data + "train_qtn_q_" + str(num_train) + ".npy", \
		train_qtn_q) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtn_q_" + str(num_val) + ".npy", \
		val_qtn_q) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtn_q_"  + str(num_test) + ".npy", \
		test_qtn_q) #Save test ground truth (100, 1096) in *.npy file
#
np.save(path_pp_data + "train_qtn_p_" + str(num_train) + ".npy", \
		train_qtn_p) #Save train ground truth (800, 1096) in *.npy file
np.save(path_pp_data + "val_qtn_p_" + str(num_val) + ".npy", \
		val_qtn_p) #Save val ground truth (100, 1096) in *.npy file
np.save(path_pp_data + "test_qtn_p_"  + str(num_test) + ".npy", \
		test_qtn_p) #Save test ground truth (100, 1096) in *.npy file
#
str_x_label = 'Training discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_q_' + str(num_train)
plot_histplot(train_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Validation discharge values' 
fig_name    = path + 'Plots/Pre_Process/val_qtn_q_' + str(num_val)
plot_histplot(val_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Testing discharge values' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_q_' + str(num_test)
plot_histplot(test_qtn_q.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_SFTMP_' + str(num_train)
plot_histplot(train_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (validation)'
fig_name    = path + 'Plots/Pre_Process/val_qtn_SFTMP_' + str(num_val)
plot_histplot(val_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'SFTMP (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_SFTMP_' + str(num_test)
plot_histplot(test_qtn_p[:,14].flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (training)' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/train_qtn_SFTMP_' + str(num_train)
plot_histplot(train_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (validation)' 
fig_name    = path + 'Plots/Pre_Process/val_qtn_SFTMP_' + str(num_val)
plot_histplot(val_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'All SWAT parameters (testing)' 
fig_name    = path + 'Plots/Pre_Process/test_qtn_SFTMP_' + str(num_test)
plot_histplot(test_qtn_p.flatten(), num_bins, label_name, str_x_label, str_y_label, fig_name, loc_pos)
#
str_x_label = 'Observational discharge values' 
str_y_label = 'Probability density'
num_bins    = 51
label_name  = 'QTN Scaler'
loc_pos     = 'upper right'
fig_name    = path + 'Plots/Pre_Process/obs_qtn_q_' + str(num_train)
plot_histplot(qq_qtn.transform(x_q_obs).flatten(), num_bins, label_name, \
				str_x_label, str_y_label, fig_name, loc_pos)
"""

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)