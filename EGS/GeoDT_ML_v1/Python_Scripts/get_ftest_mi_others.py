# Sensitivity analysis and feature importance -- v2: 13049 realz (reduced from original 46057)
#	F-test
#	Mutual information
#
# INPUTS:
#	63 inputs features
#
#OUTPUTS: 
#	hpro  -- Production Enthalpy (kJ/kg)
#	pout  -- Electrical Power Output (kW)
#	dhout -- Extracted thermal power from the injection to extraction, 
#							i.e., thermal power added to the fluid by the rock (kJ/s)
# AUTHOR: Maruti Kumar Mudunuru

import copy
import os
import pickle
import time
import scipy.stats
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#
from sklearn.feature_selection import f_regression, mutual_info_regression
#
np.set_printoptions(precision=2)

#=========================;
#  Start processing time  ;
#=========================;
tic = time.perf_counter()

#=========================================================;
#  Function-1: F-test and MI at each and every time-step  ;
#=========================================================;
def get_ftest_mi_values(num_realz, num_ts, num_params, X, y):

    #------------------;
    #  Initialization  ;
    #------------------;
    f_test_arr = np.zeros((num_params,num_ts), dtype = float) #(63,37)
    mi_arr     = np.zeros((num_params,num_ts), dtype = float) #(63,37)

    #---------------------------------------------;
    #  Calculate F-test and MI at each time-step  ;
    #---------------------------------------------;
    for i in range(1,num_ts):
    	print(i)
    	f_test, _       = f_regression(X, y[:,i]) #(63,)
    	mi              = mutual_info_regression(X, y[:,i], \
    											n_neighbors = 3, random_state = 0) #(63,)
    	f_test_arr[:,i] = copy.deepcopy(f_test) #(63,37)
    	mi_arr[:,i]     = copy.deepcopy(mi) #(63,37)

    return f_test_arr, mi_arr

#======================================================;
#  Function-2: Heatmap of F-test and MI sensitivities  ;
#======================================================;
def plot_heatmap(data_list, xticklabels, yticklabels, vmin, vmax, sq_bol, \
				cmap, str_x_label, str_y_label, str_fig_name):

    #----------------------------------------;
    #  Heatmap of the 2D sensitivity labels  ;
    #----------------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure(figsize=(35,20))
    #fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 14)
    ax = fig.add_subplot(111)
    ax = sns.heatmap(data_list, xticklabels = xticklabels, yticklabels = yticklabels, \
    				cmap = cmap, ax = ax, vmin = vmin, vmax = vmax, square = sq_bol)
    ax.invert_yaxis()
    ax.set_xlabel(str_x_label, fontsize = 48, fontweight = 'bold')
    ax.set_ylabel(str_y_label, fontsize = 48, fontweight = 'bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    #
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    #
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    #
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize = 24)
    fig.tight_layout()
    #
    ax.set_aspect(1./ax.get_data_ratio())
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')
    plt.close(fig)

#=====================================================;
#  Function-3: Plot avg. F-test and MI sensitivities  ;
#=====================================================;
def plot_avg_ftr_imp(y_pos, data_list1, data_list2, yticklabels, cmap1, cmap2, str_fig_name):

    #----------------------------------;
    #  Bar plot of avg. sensitivities  ;
    #----------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure(figsize=(35,20))
    #fig               = plt.figure()
    #
    plt.rc('text', usetex = True)
    plt.rcParams['font.family']     = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['Lucida Grande']
    plt.rc('legend', fontsize = 28)
    ax = fig.add_subplot(111)
    ax.barh(y_pos, data_list1, height = 0.4, align = 'center', color = cmap1, label = 'F-test')
    ax.barh(y_pos-0.4, data_list2, height = 0.4, align = 'center', color = cmap2, \
            alpha = 0.35, label = 'Mutual information')
    #ax.invert_yaxis()
    ax.set_xlabel('Sensitivity value', fontsize = 48, fontweight = 'bold')
    ax.set_ylabel('GeoDT features', fontsize = 48, fontweight = 'bold')
    #ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    ax.set_yticks(y_pos, labels = yticklabels)
    #
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(xaxis_format_func))
    #
    ax.tick_params(axis = 'x', which = 'major', labelsize = 40)
    ax.tick_params(axis = 'y', which = 'major', labelsize = 20)
    #
    fig.tight_layout()
    #
    ax.legend(loc='upper right')
    ax.set_aspect(1./ax.get_data_ratio())
    #fig.savefig(str_fig_name + '.pdf')
    fig.savefig(str_fig_name + '.png')
    plt.close(fig)

#*********************************************************;
#  1. Set paths, create directories, and dump .csv files  ;
#*********************************************************;
path        = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v2/'
#
df_p        = pd.read_csv(path + 'Data/geodt_params_common.csv', index_col = 0) #[13049 rows x 63 columns]
df_hpro     = pd.read_csv(path + 'Data/hpro_common.csv', index_col = 0) #[13049 rows x 37 columns]
df_pout     = pd.read_csv(path + 'Data/pout_common.csv', index_col = 0) #[13049 rows x 37 columns]
df_dhout    = pd.read_csv(path + 'Data/dhout_common.csv', index_col = 0) #[13049 rows x 37 columns]
#
num_realz   = df_hpro.shape[0] #13049
num_ts      = df_hpro.shape[1] #37
num_params  = df_p.shape[1] #63
#
hpro_geodt  = df_pout.values #(13049, 37)
pout_geodt  = df_hpro.values #(13049, 37)
dhout_geodt = df_dhout.values #(13049, 37)
#
p_geodt     = df_p.values #(13049, 63)
p_list      = df_p.columns.to_list() #63
p_list      = ['ResDepth', 'ResGradient', 'ResRho', 'ResKt', \
				'ResSv', 'AmbTempC', 'ResE', 'Resv', \
				'ResG', 'Ks3', 'Ks2', 'fNum0', 'fDia\_max0', 'fStr\_nom0', \
				'fStr\_var0', 'fDip\_nom0', 'fDip\_var0', 'fNum1', \
				'fDia\_max1', 'fStr\_nom1', 'fStr\_var1', 'fDip\_nom1', \
				'fDip\_var1', 'fNum2', 'fDia\_max2', 'fStr\_nom2', \
				'fStr\_var2', 'fDip\_nom2', 'fDip\_var2', 'bh\_bound', \
				'f\_roughness', 'w\_count', 'w\_spacing', 'w\_length', \
				'w\_azimuth', 'w\_dip', 'w\_proportion', 'w\_phase', \
				'w\_toe', 'w\_skew', 'w\_intervals', 'ra', 'rb', 'rc', \
				'dE0', 'BH\_T', 'BH\_P', 's1', 's2', 's3', 'dPp', \
				'Qinj', 'Vinj', 'Qstim', 'Vstim', 'phi0', 'phi1', 'phi2', 'mcc0', \
				'mcc1', 'mcc2', 'hfmcc', 'hfphi'] #63

#**********************************************************;
#  2a. Get F-test and MI values for each time-step (hpro)  ;
#**********************************************************;
f_test_hpro_arr, mi_hpro_arr = get_ftest_mi_values(num_realz, num_ts, num_params, p_geodt, hpro_geodt)
f_test_hpro_arr              = f_test_hpro_arr/np.max(f_test_hpro_arr) #(63,37)
mi_hpro_arr                  = mi_hpro_arr/np.max(mi_hpro_arr) #(63,37)
#
f_test_hpro                  = np.sum(f_test_hpro_arr, axis = 1)/np.max(np.sum(f_test_hpro_arr, axis = 1)) #(63,)
mi_hpro                      = np.sum(mi_hpro_arr, axis = 1)/np.max(np.sum(mi_hpro_arr, axis = 1)) #(63,)
#
#f_test_hpro                  = np.sum(f_test_hpro_arr, axis = 1)/np.sum(np.sum(f_test_hpro_arr, axis = 1)) #(63,)
#mi_hpro                      = np.sum(mi_hpro_arr, axis = 1)/np.sum(np.sum(mi_hpro_arr, axis = 1)) #(63,)
#
mi_hpro_ids                  = mi_hpro.argsort()[::-1] #Descending order of sensitivity indices
mi_hpro_sorted               = mi_hpro[mi_hpro_ids] #Descending order of avg. sensitivity values
mi_hpro_arr_sorted           = mi_hpro_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
p_sorted_mi_hpro             = [p_list[i] for i in mi_hpro_ids] #Sorted GeoDT parameter list
#
f_test_hpro_arr_sorted       = f_test_hpro_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
f_test_hpro_sorted           = f_test_hpro[mi_hpro_ids] #Descending order of avg. sensitivity values

#***************************************************;
#  2b. Get f-test and mi for each time-step (pout)  ;
#***************************************************;
f_test_pout_arr, mi_pout_arr = get_ftest_mi_values(num_realz, num_ts, num_params, p_geodt, pout_geodt)
f_test_pout_arr              = f_test_pout_arr/np.max(f_test_pout_arr) #(63,37)
mi_pout_arr                  = mi_pout_arr/np.max(mi_pout_arr) #(63,37)
#
f_test_pout                  = np.sum(f_test_pout_arr, axis = 1)/np.max(np.sum(f_test_pout_arr, axis = 1)) #(63,)
mi_pout                      = np.sum(mi_pout_arr, axis = 1)/np.max(np.sum(mi_pout_arr, axis = 1)) #(63,)
#
#f_test_pout                  = np.sum(f_test_pout_arr, axis = 1)/np.sum(np.sum(f_test_pout_arr, axis = 1)) #(63,)
#mi_pout                      = np.sum(mi_pout_arr, axis = 1)/np.sum(np.sum(mi_pout_arr, axis = 1)) #(63,)
#
mi_pout_sorted               = mi_pout[mi_hpro_ids] #Descending order of avg. sensitivity values
mi_pout_arr_sorted           = mi_pout_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
p_sorted_mi_pout             = [p_list[i] for i in mi_hpro_ids] #Sorted GeoDT parameter list
#
f_test_pout_arr_sorted       = f_test_pout_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
f_test_pout_sorted           = f_test_pout[mi_hpro_ids] #Descending order of avg. sensitivity values

#****************************************************;
#  2c. Get f-test and mi for each time-step (dhout)  ;
#****************************************************;
f_test_dhout_arr, mi_dhout_arr = get_ftest_mi_values(num_realz, num_ts, num_params, p_geodt, dhout_geodt)
f_test_dhout_arr              = f_test_dhout_arr/np.max(f_test_dhout_arr) #(63,37)
mi_dhout_arr                  = mi_dhout_arr/np.max(mi_dhout_arr) #(63,37)
#
f_test_dhout                  = np.sum(f_test_dhout_arr, axis = 1)/np.max(np.sum(f_test_dhout_arr, axis = 1)) #(63,)
mi_dhout                      = np.sum(mi_dhout_arr, axis = 1)/np.max(np.sum(mi_dhout_arr, axis = 1)) #(63,)
#
#f_test_dhout                  = np.sum(f_test_dhout_arr, axis = 1)/np.sum(np.sum(f_test_dhout_arr, axis = 1)) #(63,)
#mi_dhout                      = np.sum(mi_dhout_arr, axis = 1)/np.sum(np.sum(mi_dhout_arr, axis = 1)) #(63,)
#
mi_dhout_sorted               = mi_dhout[mi_hpro_ids] #Descending order of avg. sensitivity values
mi_dhout_arr_sorted           = mi_dhout_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
p_sorted_mi_dhout             = [p_list[i] for i in mi_hpro_ids] #Sorted GeoDT parameter list
#
f_test_dhout_arr_sorted       = f_test_dhout_arr[mi_hpro_ids,:] #Descending order of time-step sensitivity values
f_test_dhout_sorted           = f_test_dhout[mi_hpro_ids] #Descending order of avg. sensitivity values

#******************************************;
#  3a. Plot F-test and MI heatmaps (hpro)  ;
#******************************************;
str_x_label   = 'Time-steps'
str_y_label   = 'GeoDT features'
str_fig_name  = path + 'Plots/mi_hpro_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Blues"
#hm_labels     = copy.deepcopy(mi_hpro_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(mi_hpro_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
				cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_hpro_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Purples"
#hm_labels     = copy.deepcopy(f_test_hpro_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(f_test_hpro_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
                cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_mi_hpro_avg'
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap1         = "purple"
cmap2         = "blue"
y_pos         = np.arange(len(p_list))
#
plot_avg_ftr_imp(y_pos, f_test_hpro_sorted, mi_hpro_sorted, yticklabels, cmap1, cmap2, str_fig_name)

#******************************************;
#  3b. Plot F-test and MI heatmaps (pout)  ;
#******************************************;
str_x_label   = 'Time-steps'
str_y_label   = 'GeoDT features'
str_fig_name  = path + 'Plots/mi_pout_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Blues"
#hm_labels     = copy.deepcopy(mi_pout_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(mi_pout_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
                cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_pout_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Purples"
#hm_labels     = copy.deepcopy(f_test_pout_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(f_test_pout_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
                cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_mi_pout_avg'
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap1         = "purple"
cmap2         = "blue"
y_pos         = np.arange(len(p_list))
#
plot_avg_ftr_imp(y_pos, f_test_pout_sorted, mi_pout_sorted, yticklabels, cmap1, cmap2, str_fig_name)

#*******************************************;
#  3c. Plot F-test and MI heatmaps (dhout)  ;
#*******************************************;
str_x_label   = 'Time-steps'
str_y_label   = 'GeoDT features'
str_fig_name  = path + 'Plots/mi_dhout_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Blues"
#hm_labels     = copy.deepcopy(mi_dhout_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(mi_dhout_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
                cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_dhout_ts_sorted'
xticklabels   = 1
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap          = "Purples"
#hm_labels     = copy.deepcopy(f_test_dhout_arr[0:10,0:5]) #To get a sense of matrix mapping plot
hm_labels     = copy.deepcopy(f_test_dhout_arr_sorted)
sq_bol        = True
vmin          = np.min(hm_labels)
vmax          = np.max(hm_labels)
#
plot_heatmap(hm_labels, xticklabels, yticklabels, vmin, vmax, sq_bol, \
                cmap, str_x_label, str_y_label, str_fig_name)
#
str_fig_name  = path + 'Plots/ftest_mi_dhout_avg'
yticklabels   = copy.deepcopy(p_sorted_mi_hpro)
cmap1         = "purple"
cmap2         = "blue"
y_pos         = np.arange(len(p_list))
#
plot_avg_ftr_imp(y_pos, f_test_dhout_sorted, mi_dhout_sorted, yticklabels, cmap1, cmap2, str_fig_name)


#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)