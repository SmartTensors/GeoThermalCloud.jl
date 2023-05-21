# Sensitivity analysis and feature importance -- 4078 realz 
#	F-test
#	Mutual information
#
# INPUTS:
#	117 inputs features
#
#OUTPUTS: 
#   NPV   -- Net present value
#
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

#=====================================================;
#  Function-1: Plot avg. F-test and MI sensitivities  ;
#=====================================================;
def plot_avg_ftr_imp(y_pos, data_list1, data_list2, yticklabels, cmap1, cmap2, str_fig_name):

    #----------------------------------;
    #  Bar plot of avg. sensitivities  ;
    #----------------------------------;
    legend_properties = {'weight':'bold'}
    fig               = plt.figure(figsize=(35,25))
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
    ax.tick_params(axis = 'y', which = 'major', labelsize = 18)
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
path        = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
path_geodt  = path + 'Data/' #GeoDT data
#
df_p        = pd.read_csv(path + 'Data/geodt_params.csv', index_col = 0) #[4078 rows x 117 columns]
df_npv      = pd.read_csv(path_geodt + 'npv.csv', index_col = 0) #[4078 rows x 1 columns]
#
num_realz   = df_npv.shape[0] #4078
num_params  = df_p.shape[1] #117
#
npv_geodt   = df_npv.values #(4078, 1)
#
p_geodt     = df_p.values #(4078, 117)
p_list      = df_p.columns.to_list() #117
p_list      = ['pin', 'size', 'ResDepth', 'ResGradient', 'ResRho', 'ResKt', 'ResSv', 'AmbTempC',
                'AmbPres', 'ResE', 'Resv', 'ResG', 'Ks3', 'Ks2', 's3Azn', 's3AznVar', 's3Dip', 
                's3DipVar', 'fNum0', 'fDia\_min0', 'fDia\_max0', 'fStr\_nom0', 'fStr\_var0', 
                'fDip\_nom0', 'fDip\_var0', 'fNum1', 'fDia\_min1', 'fDia\_max1', 'fStr\_nom1', 
                'fStr\_var1', 'fDip\_nom1', 'fDip\_var1', 'fNum2', 'fDia\_min2', 'fDia\_max2', 
                'fStr\_nom2', 'fStr\_var2', 'fDip\_nom2', 'fDip\_var2', 'alpha0', 'alpha1', 
                'alpha2', 'gamma0', 'gamma1', 'gamma2', 'n10', 'n11', 'n12', 'a0', 'a1',
                'a2', 'b0', 'b1', 'b2', 'N0', 'N1', 'N2', 'bh0', 'bh1', 'bh2', 'bh\_min', 
                'bh\_max', 'bh\_bound', 'f\_roughness', 'w\_count', 'w\_spacing', 'w\_length', 
                'w\_azimuth', 'w\_dip', 'w\_proportion', 'w\_phase', 'w\_toe', 'w\_skew', 
                'w\_intervals', 'ra', 'rb', 'rc', 'rgh', 'CemKt', 'CemSv', 'GenEfficiency', 
                'LifeSpan', 'TimeSteps', 'p\_whp', 'Tinj', 'H\_ConvCoef', 'dT0', 'dE0',
                'PoreRho', 'Poremu', 'Porek', 'Frack', 'BH\_T', 'BH\_P', 's1', 's2',
                's3', 'perf', 'r\_perf', 'sand', 'leakoff', 'dPp', 'dPi', 'stim\_limit', 
                'Qinj', 'Vinj', 'Qstim', 'Vstim', 'bval', 'phi0', 'phi1', 'phi2', 'mcc0', 
                'mcc1', 'mcc2', 'hfmcc', 'hfphi'] #117

#***************************************;
#  2. Get F-test and MI values for NPV  ;
#***************************************;
f_test, _            = f_regression(p_geodt, npv_geodt.flatten()) #(117,)
mi_npv               = mutual_info_regression(p_geodt, npv_geodt, n_neighbors = 3, random_state = 0) #(117,)
ind_list             = np.argwhere(np.isnan(f_test))[:,0]
f_test_npv           = copy.deepcopy(f_test)
f_test_npv[ind_list] = 0
#
f_test_npv           = f_test_npv/np.max(f_test_npv) #(117,)
mi_npv               = mi_npv/np.max(mi_npv) #(117,)

mi_npv_ids           = mi_npv.argsort()[::-1] #Descending order of sensitivity indices
mi_npv_sorted        = mi_npv[mi_npv_ids] #Descending order of avg. sensitivity values
p_sorted_mi_npv      = [p_list[i] for i in mi_npv_ids] #Sorted GeoDT parameter list
#
f_test_npv_sorted    = f_test_npv[mi_npv_ids] #Descending order of avg. sensitivity values

#****************************************;
#  3. Plot F-test and MI heatmaps (NPV)  ;
#****************************************;
str_x_label   = 'Time-steps'
str_y_label   = 'GeoDT features'
str_fig_name  = path + 'Plots/ftest_mi_npv_avg'
yticklabels   = copy.deepcopy(p_sorted_mi_npv)
cmap1         = "purple"
cmap2         = "blue"
y_pos         = np.arange(len(p_list))
#
plot_avg_ftr_imp(y_pos, f_test_npv_sorted, mi_npv_sorted, yticklabels, cmap1, cmap2, str_fig_name)

#======================;
# End processing time  ;
#======================;
toc = time.perf_counter()
print('Time elapsed in seconds = ', toc - tic)