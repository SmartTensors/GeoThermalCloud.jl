# mpi4py ON TH -- Train 21875 models
#
# Please run 'python get_dir_hp_dnn.py' before doing anything below
# mpirun -n 4 python get_dnn_results_mpi4py.py >> dnn_results.txt (if using mpi4py on MacOSX)
#
#    1. DNN-based inverse model to train/validate/test the proposed framework
#    3. INPUTS are GeoDT params
#       (a) Pre-processing based on StandardScaler for geodt-params-data
#    4. OUTPUTS are NPV
#       (a) Pre-processing based on StandardScaler NPV-data
#    5. n_realz = 4078, Train/Val/Test = 80/10/10
#
# mpirun -n 50 get_dnn_results_mpi4py.py >> dnn_results.txt #PNNL's pinklady
# srun -n 32 -c 2 --cpu-bind=cores python get_dnn_results_mpi4py.py >> hp_output_ss_nersc.txt #Haswell
# srun -n 68 -c 4 --cpu-bind=cores python get_dnn_results_mpi4py.py >> hp_output_ss_nersc.txt #KNL
#
# srun -n 21875 -c 4 --cpu-bind=cores python get_dnn_results_mpi4py.py >> hp_output_ss_nersc.txt #KNL
# killall python
#
# MAKESURE THESE ARE LOADED ON NERSC before launching the job:
# 	module load texlive
# 	module load python/3.9-anaconda-2021.11
# 	module load cray-hdf5-parallel
# 	conda activate /global/common/software/m1800/maruti_python/conda/myenv1
# 	export HDF5_USE_FILE_LOCKING=FALSE
#
# ls | wc -l
# AUTHOR: Maruti Kumar Mudunuru
# https://stackoverflow.com/questions/51629763/python-multiprocessing-google-compute-engine
# https://serverfault.com/questions/1033159/running-a-task-in-parallel-in-multiple-machines-in-gcp-and-orchestrating-it

from mpi4py import MPI
#
import os
import copy
import time
import yaml
import pydot
import graphviz
import pydotplus
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
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
#
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import CSVLogger

#=======================================================;
#  Function-1: GeoDT params to NPV (Forward-DNN-model)  ;
#=======================================================;
def get_dnn_model(geodt_out_comps, geodt_p_comps, \
                    loop_units, dense_units, alpha_units, \
                    dropout_units):

    #--------------------------------------;
    #  Construct DNN based on hp.txt file  ;
    #--------------------------------------;
    inp_shape      = (geodt_p_comps,) #117
    #
    input_layer    = Input(shape = inp_shape, name = "GeoDT-params-in") #117
    x              = input_layer
    #  
    for i in range(loop_units):
        x = Dense(units = dense_units[i], name = "Dense-" + str(i))(x)
        x = LeakyReLU(alpha = alpha_units, name = "Activation-" + str(i))(x)
        x = Dropout(dropout_units)(x)

    out   = Dense(units = geodt_out_comps, name = "GeoDT-npv-out")(x) #1
    model = Model(input_layer, out, name = "Forward-GeoDT-p-to-npv-Model")

    return model

#=========================================================;
#  Function-2: Plot training and validation loss ('mse')  ; 
#=========================================================;
def plot_tv_loss(hist, epochs, path_fl_sav):

	#---------------------;
	#  Plot loss ('mse')  ;
	#---------------------;
	legend_properties = {'weight':'bold'}
	fig               = plt.figure()
	#
	#plt.rc('text', usetex = True)
	plt.rcParams['font.family']     = ['sans-serif']
	plt.rcParams['font.sans-serif'] = ['Lucida Grande']
	plt.rc('legend', fontsize = 14)
	ax = fig.add_subplot(111)
	ax.set_xlabel('Epoch', fontsize = 24, fontweight = 'bold')
	ax.set_ylabel('Loss (MSE)', fontsize = 24, fontweight = 'bold')
	#plt.grid(True)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
	ax.set_xlim([0, epochs])
	#ax.set_ylim([0.35, 1])
	e_list = [i for i in range(0,epochs)]
	ax.plot(e_list, hist['loss'], linestyle = 'solid', linewidth = 1.5, \
					color = 'b', label = 'Training') #Training loss
	ax.plot(e_list, hist['val_loss'], linestyle = 'solid', linewidth = 1.5, \
					color = 'm', label = 'Validation') #Validation loss
	#tick_spacing = 100
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	ax.legend(loc = 'upper right')
	fig.tight_layout()
	#fig.savefig(path_fl_sav + 'Loss.pdf')
	fig.savefig(path_fl_sav + 'Loss.png')
	plt.close(fig)

#======================================================================;
#  Function-3: Plot one-to-one for train/val/test (All GeoDT outputs)  ; 
#======================================================================;
def plot_gt_pred(x, y, param_id, fl_name, str_x_label, str_y_label):

	#------------------------------------------------;
	#  Plot one-to-one (ground truth vs. predicted)  ;
	#------------------------------------------------;
	legend_properties = {'weight':'bold'}
	fig               = plt.figure()
	#
	#plt.rc('text', usetex = True)   
	plt.rcParams['font.family']     = ['sans-serif']
	plt.rcParams['font.sans-serif'] = ['Lucida Grande']
	plt.rc('legend', fontsize = 14)
	ax = fig.add_subplot(111)
	ax.set_xlabel(str_x_label, fontsize = 14, fontweight = 'bold')
	ax.set_ylabel(str_y_label, fontsize = 14, fontweight = 'bold')
	#plt.grid(True)
	ax.tick_params(axis = 'both', which = 'major', labelsize = 20, length = 6, width = 2)
	min_val = np.min(x)
	max_val = np.max(x)
	ax.set_xlim([min_val, max_val])
	ax.set_ylim([min_val, max_val])
	ax.plot([min_val, max_val], [min_val, max_val], \
					linestyle = 'solid', linewidth = 1.5, \
					color = 'r') #One-to-One line
	ax.scatter(x, y, color = 'b', marker = 'o')
	#tick_spacing = 100
	#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	#ax.legend(loc = 'upper right')
	ax.set_aspect(1./ax.get_data_ratio())
	fig.tight_layout()
	#fig.savefig(fl_name + str(param_id) + '.pdf')
	fig.savefig(fl_name + str(param_id) + '.png')
	plt.close(fig)

#====================================================================;
#  Function-4: Train individual models (mpi4py calls this function)  ; 
#====================================================================;
def get_trained_models(k_child_rank, start_at_this_hpfolder, random_seed):

	#-------------------;
	#  0. Get realz_id  ;
	#-------------------;
	counter = k_child_rank + start_at_this_hpfolder - 1
	#
	#counter_list  = [1, 9376, 18751]
	#counter       = counter_list[0]

	#------------------------------------------------;
	#  1. Get pre-processed data (all realizations)  ;
	#------------------------------------------------;
	#path = os.getcwd() #Get current directory path
	#path           = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
	#path           = '/home/mudu605/2_GeoDT_DL/'
	#path           = '/mnt/4tba/maruti/11_GeoDT_DL/'
	path           = '/tahoma/emsle60558/test_dl_1/1_ML4GeoDT_v3/'
	path_geodt     = path + 'Data/' #GeoDT data
	path_ind       = path + 'Data/Train_Val_Test_Indices/' #Train/Val/Test indices
	path_pp_models = path + 'Data/PreProcess_Models/' #Pre-processing models for standardization
	path_pp_data   = path + 'Data/PreProcessed_Data/' #Pre-processed data
	path_raw_data  = path + 'Data/Raw_Data/' #Raw data
	#
	#path_testing   = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/' #21875 models and their inputs are here
	#path_testing   = '/home/mudu605/2_GeoDT_DL/'
	#path_testing    = '/mnt/4tba/maruti/11_GeoDT_DL/'
	path_testing   = '/tahoma/emsle60558/test_dl_1/1_ML4GeoDT_v3/'
	#
	path_fl_sav    =  path_testing + "1_InvDNNModel_ss_th/" + str(counter) + "_model/" #i-th hp-dl-model folder
	print(path_fl_sav + 'hp_input_deck.txt')

	#--------------------------------;
	#  2. Get other initializations  ;
	#--------------------------------;
	num_realz       = 4078 #No. of realization (total realz data)
	num_train       = 3278 #Training realz
	num_val         = 400 #Validation realz
	num_test        = 400 #Testing realz
	#
	sclr_name       = "ss" #Standard Scaler
	geodt_out_list  = ['npv_']
	geodt_out       = geodt_out_list[0]
	p_name          = path_pp_models + "p_" + sclr_name + "_" + str(num_train) + ".sav" #GeoDT-params pre-processor
	q_name          = path_pp_models + geodt_out + sclr_name + "_" + str(num_train) + ".sav" #q-data pre-processor
	#
	pp_scalar       = pickle.load(open(p_name, 'rb')) #Load already created GeoDT pre-processing model
	qq_scalar       = pickle.load(open(q_name, 'rb')) #Load already created q-data pre-processing model
	#
	df_p            = pd.read_csv(path_geodt + 'geodt_params.csv', index_col = 0) #[4078 rows x 117 columns]

	#--------------------------------------------------------------------;
	#  3a. Load train/val/test *.npy data -- One of the 7-preprocessors  ;
	#--------------------------------------------------------------------; 
	train_ss_npv = np.load(path_pp_data + "train_" + sclr_name + "_" + geodt_out + \
						str(num_train) + ".npy") #Train q-data pre-processed (3278, 1) in *.npy file
	val_ss_npv   = np.load(path_pp_data + "val_" + sclr_name + "_" + geodt_out + \
						str(num_val) + ".npy") #Save q-data pre-processed (400, 1) in *.npy file
	test_ss_npv  = np.load(path_pp_data + "test_" + sclr_name + "_" + geodt_out  + \
						str(num_test) + ".npy") #Save q-data pre-processed (400, 1) in *.npy file 
	#
	train_ss_p   = np.load(path_pp_data + "train_" + sclr_name + "_p_" + \
						str(num_train) + ".npy") #Train p-data pre-processed (3278, 117) in *.npy file
	val_ss_p     = np.load(path_pp_data + "val_" + sclr_name + "_p_" + \
						str(num_val) + ".npy") #Save p-data pre-processed (400, 117) in *.npy file
	test_ss_p    = np.load(path_pp_data + "test_" + sclr_name + "_p_"  + \
						str(num_test) + ".npy") #Save p-data pre-processed (400, 117) in *.npy file

	#--------------------------------------------------;
	#  3b. Only use non-outlier data (Train/Val/Test)  ;
	#      (NPV > -1)                                  ;
	#--------------------------------------------------; 
	counterx = 0
	train_neglect_list = []
	#
	for i in train_ss_npv: #Neglect train-counterx = 0 to 13
	    if i <= -1:
	        #print('Train = ', counterx, i)
	        train_neglect_list.append(counterx)
	    counterx = counterx + 1

	counterx = 0
	val_neglect_list = []
	#
	for i in val_ss_npv: #Neglect val-counterx = 0
	    if i <= -1:
	        #print('Val = ', counterx, i)
	        val_neglect_list.append(counterx)
	    counterx = counterx + 1

	counterx = 0
	test_neglect_list = []
	#
	for i in test_ss_npv: #Neglect test-counterx = 0 to 2
	    if i <= -1:
	        #print('Test = ', counterx, i)
	        test_neglect_list.append(counterx)
	    counterx = counterx + 1

	train_nn_list = np.array(list(set(range(0,num_train)) - set(train_neglect_list)), dtype = int) #(3264,)
	val_nn_list   = np.array(list(set(range(0,num_val)) - set(val_neglect_list)), dtype = int) #(399,)
	test_nn_list  = np.array(list(set(range(0,num_test)) - set(test_neglect_list)), dtype = int) #(397,)
	#
	train_q       = train_ss_npv[train_nn_list,:] #(3264,1)
	val_q         = val_ss_npv[val_nn_list,:] #(399,1)
	test_q        = test_ss_npv[test_nn_list,:] #(397,1)
	#
	train_p       = train_ss_p[train_nn_list,:] #(3264,117)
	val_p         = val_ss_p[val_nn_list,:] #(399,117)
	test_p        = test_ss_p[test_nn_list,:] #(397,117)

	#----------------------------------------------;
	#  4a. Model training initialization           ;
	#      (read from a hyperparameter .txt file)  ;
	#----------------------------------------------;
	K.clear_session()
	#
	np.random.seed(random_seed)
	#
	nq_comps      = train_q.shape[1] #1
	np_comps      = train_p.shape[1] #117
	#
	fl_id         = open(path_fl_sav + 'hp_input_deck.txt') #Read the hp .txt file line by line
	#
	hp_line_list  = fl_id.readlines()
	#
	num_layers    = int(hp_line_list[0].strip().split(" = ", 1)[1])
	neurons       = [int(i) for i in hp_line_list[1].strip().split(" = ", 1)[1].split(",")]
	dropout_value = float(hp_line_list[2].strip().split(" = ", 1)[1])
	alpha_value   = float(hp_line_list[3].strip().split(" = ", 1)[1])
	lr_values     = float(hp_line_list[4].strip().split(" = ", 1)[1])
	epochs        = int(hp_line_list[5].strip().split(" = ", 1)[1])
	batch_size    = int(hp_line_list[6].strip().split(" = ", 1)[1])
	#
	print(num_layers, neurons, dropout_value, alpha_value, lr_values, epochs, batch_size)
	#
	for hp_line in hp_line_list:
		temp = hp_line.strip().split(" = ", 1)
		print(temp, len(temp))
	fl_id.close()
	
	#-------------------------------------;
	#  4b. Model training and validation  ;
	#-------------------------------------; 
	fwd_model  = get_dnn_model(nq_comps, np_comps, num_layers, \
								neurons, alpha_value, dropout_value) #Forward-DNN-model
	fwd_model.summary() #Model summary
	tf.keras.utils.plot_model(fwd_model, path_fl_sav + "Full-Fwd-DNN-Model-a.png", show_shapes = True)
	tf.keras.utils.plot_model(fwd_model, path_fl_sav + "Full-Fwd-DNN-Model-b.png", show_shapes = False)
	#
	opt        = Adam(learning_rate = lr_values) #Optimizer and learning rate
	loss       = "mse" #MSE loss function
	fwd_model.compile(opt, loss = loss)
	train_csv  = path_fl_sav + "FwdDNNModel_Loss.csv"
	csv_logger = CSVLogger(train_csv)
	callbacks  = [csv_logger] 
    #
	history    = fwd_model.fit(x = train_p, y = train_q, \
								epochs = epochs, batch_size = batch_size, \
								validation_data = (val_p, val_q), \
								verbose = 2, callbacks = callbacks)
	hist = history.history
	print("Done training")
	#print(hist.keys())
	time.sleep(2)

	#--------------------------------------;
	#  5. Plot train and val loss ('mse')  ;
	#     (loss and epoch stats)           ;
	#--------------------------------------;
	plot_tv_loss(hist, epochs, path_fl_sav)
	#
	df_hist        = pd.read_csv(train_csv)
	val_f1         = df_hist['val_loss']
	min_val_f1     = val_f1.min()
	min_val_f1_df  = df_hist[val_f1 == min_val_f1]
	min_epochs     = min_val_f1_df['epoch'].values
	min_val_loss   = min_val_f1_df['val_loss'].values
	min_train_loss = min_val_f1_df['loss'].values
	#
	print(min_val_f1_df)

	#----------------------------------------;
	#  6. Model prediction (train/val/test)  ;
	#----------------------------------------;
	train_pred_q = fwd_model.predict(train_p)
	val_pred_q   = fwd_model.predict(val_p)
	test_pred_q  = fwd_model.predict(test_p)
	#
	#np.save(path_fl_sav + "train_pred_q_" + str(num_train) + ".npy", \
	#	train_pred_q) #Save train pred (3278, 1) in *.npy file (normalized)
	#np.save(path_fl_sav + "val_pred_q_" + str(num_val) + ".npy", \
	#	val_pred_q) #Save val pred (400, 1) in *.npy file (normalized)
	#np.save(path_fl_sav + "test_pred_q_" + str(num_test) + ".npy", \
	#	test_pred_q) #Save test pred (400, 1) in *.npy file (normalized)

	#-----------------------------------------------------;
	#  7. Inverse transformed GeoDT npv (train/val/test)  ;
	#-----------------------------------------------------;
	train_q_it      = qq_scalar.inverse_transform(train_q)
	val_q_it        = qq_scalar.inverse_transform(val_q)
	test_q_it       = qq_scalar.inverse_transform(test_q)
	#
	train_pred_q_it = qq_scalar.inverse_transform(train_pred_q)
	val_pred_q_it   = qq_scalar.inverse_transform(val_pred_q)
	test_pred_q_it  = qq_scalar.inverse_transform(test_pred_q)
	#
	#np.save(path_fl_sav + "train_pred_q_it" + str(num_train) + ".npy", \
	#	train_pred_q_it) #Save train pred (3278, 1) in *.npy file
	#np.save(path_fl_sav + "val_pred_q_it" + str(num_val) + ".npy", \
	#	val_pred_q_it) #Save val pred (400, 1) in *.npy file
	#np.save(path_fl_sav + "test_pred_q_it" + str(num_test) + ".npy", \
	#	test_pred_q_it) #Save test pred (400, 1) in *.npy file

	#---------------------------------------------------;
	#  8. One-to_one normalized plots (train/val/test)  ;
	#---------------------------------------------------;
	str_x_label = 'Normalized ground truth (NPV)'
	str_y_label = 'Normalized prediction (NPV)'
	#
	param_id = 0
	x_train  = train_q[:,param_id]
	y_train  = train_pred_q[:,param_id]
	fl_name  = path_fl_sav + "Q_train_"
	plot_gt_pred(x_train, y_train, param_id, fl_name, \
					str_x_label, str_y_label)
	#  
	x_val    = val_q[:,param_id]
	y_val    = val_pred_q[:,param_id]
	fl_name  = path_fl_sav + "Q_val_"
	plot_gt_pred(x_val, y_val, param_id, fl_name, \
					str_x_label, str_y_label)
	#
	x_test   = test_q[:,param_id]
	y_test   = test_pred_q[:,param_id]
	fl_name  = path_fl_sav + "Q_test_"
	plot_gt_pred(x_test, y_test, param_id, fl_name, \
					str_x_label, str_y_label)
	
	#----------------------------------------------------------;
	#  9. One-to-one inverse transform plots (train/val/test)  ;
	#----------------------------------------------------------;
	str_x_label = 'Ground truth (NPV)'
	str_y_label = 'Prediction (NPV)'
	#
	param_id = 0
	x_train  = train_q_it[:,param_id]
	y_train  = train_pred_q_it[:,param_id]
	fl_name  = path_fl_sav + "IT_train_"
	plot_gt_pred(x_train, y_train, param_id, fl_name, \
					str_x_label, str_y_label)
	#  
	x_val    = val_q_it[:,param_id]
	y_val    = val_pred_q_it[:,param_id]
	fl_name  = path_fl_sav + "IT_val_"
	plot_gt_pred(x_val, y_val, param_id, fl_name, \
					str_x_label, str_y_label)
	#
	x_test   = test_q_it[:,param_id]
	y_test   = test_pred_q_it[:,param_id]
	fl_name  = path_fl_sav + "IT_test_"
	plot_gt_pred(x_test, y_test, param_id, fl_name, \
					str_x_label, str_y_label)

	#--------------------------------------------------------------;
	#  10. Save model (TensorFlow SavedModel format. *.h5 format)  ;
	#--------------------------------------------------------------;
	#fwd_model.save(path_fl_sav + "Fwd_DNN_Model") #TensorFlow SavedModel format
	#fwd_model.save(path_fl_sav + "Fwd_DNN_Model.h5") #h5 format

	#-------------------------------------------------------------;
	#  12. Done training and saving the DL model and its outputs  ;
	#-------------------------------------------------------------;
	print('-------------------------------------------------------------')
	print('            Trained: ' + str(counter) + '_model/            ',)
	print('-------------------------------------------------------------')

#**************************************************;
#  mpi4py + TFv2 + ParallelHDF5 for runs on NERSC  ;
#**************************************************;
#
if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
	tic = time.perf_counter()

	#=======================================;
	#  2. MPI communicator, size, and rank  ;
	#=======================================;
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()

	#============================================;
	#  3. Number of realz per process/rank/core  ;
	#============================================;
	num_total = 20 #Total number of realization that needs to run on a given process/rank/core

	#=========================================;
	#  4. MPI send and receive realz numbers  ;
	#=========================================;
	if rank == 0:
		for i in range(size-1,-1,-1):
			realz_id = [0]*num_total  #Realization list
			#
			for j in range(num_total):
				print(j + num_total*i + 1, realz_id)
				realz_id[j] = j + num_total*i + 1
			print('rank and realz_id = ', rank, realz_id)
			#
			if i > 0: 
				comm.send(realz_id, dest = i)
	else:
		realz_id = comm.recv(source = 0)
		print('rank, realz_id = ', rank, realz_id)

	#==========================================;
	#  5. Run DL model training for each realz ;
	#==========================================;
	for k in realz_id:
		start_at_this_hpfolder = 1 #1, 9376, 18751 (1, 2, and 3-DNN layers)
		random_seed            = 1337
		get_trained_models(k, start_at_this_hpfolder, random_seed)
		print('rank, k, realz_id = ', rank, k, realz_id, k+start_at_this_hpfolder-1)

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)