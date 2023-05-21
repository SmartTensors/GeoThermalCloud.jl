# Create directories and hyperparameter input files for DNN-based forward models
# (Create directories on UBUNTU)
# Total number of DNN models trained = 21875
#  Sanity check:
#	https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module
#
# module load texlive
# module load python/3.9-anaconda-2021.11
# module load cray-hdf5-parallel
# conda activate /global/common/software/m1800/maruti_python/conda/myenv1
# export HDF5_USE_FILE_LOCKING=FALSE
#
# python get_dir_hp_dnn_we.py
# AUTHOR: Maruti Kumar Mudunuru

import numpy as np
import glob
import os
import time
import subprocess
import itertools

if __name__ == '__main__':

	#=========================;
	#  Start processing time  ;
	#=========================;
	tic = time.perf_counter()

	#----------------------------;
	#  1. Paths for directories  ;
	#----------------------------;
	#path           = '/Users/mudu605/Desktop/GeoDT_DL/1_ML4GeoDT_v3/'
	path           = '/home/mudu605/2_GeoDT_DL/'
	#path            = '/mnt/4tba/maruti/11_GeoDT_DL/'
	dir_path = path + "1_InvDNNModel_ss_we/"
	print(dir_path)
	#
	if not os.path.exists(dir_path): #Create if they dont exist
		os.makedirs(dir_path)

	num_dnn_folders = 21875
	#
	for i in range(1,num_dnn_folders+1):
		dir_path    = path + "1_InvDNNModel_ss_we/" + str(i) + "_model/"
		print(dir_path)
		#
		if not os.path.exists(dir_path): #Create if they dont exist
			os.makedirs(dir_path)

	#---------------------------------------------------------;
	#  2a. Create hyperparameters (1-DNN-layer): 9375 models  ;
	#---------------------------------------------------------;
	num_models         = 9375 # 3*3125
	num_layers         = 1 #number of hidden layers 
	neurons_list       = [1000, 500, 250] #3
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	alpha_value_list   = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [100, 200, 300, 400, 500] #5
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 1
	hp_1dnn_list = [] #9375
	#
	for neurons in neurons_list:
		for dropout_value in dropout_value_list:
			for alpha_value in alpha_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_1dnn_list.append([counter, \
													num_layers, \
													neurons, \
													dropout_value, \
													alpha_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, neurons, \
		dropout_value, alpha_value, \
		lr_values, epochs, batch_size = hp_1dnn_list[i]
		#
		path_fl_sav = path + "1_InvDNNModel_ss_we/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('neurons       = ' + str(neurons))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('alpha_value   = ' + str(alpha_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#---------------------------------------------------------;
	#  2b. Create hyperparameters (2-DNN-layer): 9375 models  ;
	#---------------------------------------------------------;
	num_models         = 9375 # 3*3125
	num_layers         = 2 #number of hidden layers 
	neurons_list       = [[1000, 500], \
							[1000, 250], \
							[500, 250]] #3
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	alpha_value_list   = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [100, 200, 300, 400, 500] #5
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 9376
	hp_2dnn_list = [] #9375
	#
	for neurons in neurons_list:
		for dropout_value in dropout_value_list:
			for alpha_value in alpha_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_2dnn_list.append([counter, \
													num_layers, \
													neurons, \
													dropout_value, \
													alpha_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, neurons, \
		dropout_value, alpha_value, \
		lr_values, epochs, batch_size = hp_2dnn_list[i]
		#
		path_fl_sav = path + "1_InvDNNModel_ss_we/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('neurons       = ' + str(neurons[0]) + ', ' + str(neurons[1]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('alpha_value   = ' + str(alpha_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#---------------------------------------------------------;
	#  2c. Create hyperparameters (3-DNN-layer): 3125 models  ;
	#---------------------------------------------------------;
	num_models         = 3125 # 3125
	num_layers         = 3 #number of hidden layers 
	neurons_list       = [[1000, 500, 250]] #1
	dropout_value_list = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	alpha_value_list   = [0.0, 0.1, 0.2, 0.3, 0.4] #5
	lr_values_list     = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] #5
	epochs_list        = [100, 200, 300, 400, 500] #5
	batch_sizes_list   = [4, 8, 16, 32, 64] #5
	#
	counter      = 18751
	hp_3dnn_list = [] #3125
	#
	for neurons in neurons_list:
		for dropout_value in dropout_value_list:
			for alpha_value in alpha_value_list:
				for lr_values in lr_values_list:
					for epochs in epochs_list:
						for batch_size in batch_sizes_list:
							hp_3dnn_list.append([counter, \
													num_layers, \
													neurons, \
													dropout_value, \
													alpha_value, \
													lr_values, \
													epochs, \
													batch_size])
							print(counter)
							counter = counter + 1

	for i in range(0,num_models):
		counter, num_layers, neurons, \
		dropout_value, alpha_value, \
		lr_values, epochs, batch_size = hp_3dnn_list[i]
		#
		path_fl_sav = path + "1_InvDNNModel_ss_we/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		f = open(path_fl_sav,'w+')
		f.write('num_layers    = ' + str(num_layers))
		f.write('\n')
		f.write('neurons       = ' + str(neurons[0]) + ', ' + str(neurons[1]) \
									+ ', ' + str(neurons[2]))
		f.write('\n')
		f.write('dropout_value = ' + str(dropout_value))
		f.write('\n')
		f.write('alpha_value   = ' + str(alpha_value))
		f.write('\n')
		f.write('lr_values     = ' + str(lr_values))
		f.write('\n')
		f.write('epochs        = ' + str(epochs))
		f.write('\n')
		f.write('batch_size    = ' + str(batch_size))
		f.close()

	#--------------------------;
	#  3. Test some scenarios  ;
	#--------------------------;
	counter_list  = [1, 9376, 18751]
	#
	for counter in counter_list:
		path_fl_sav   = path + "1_InvDNNModel_ss_we/" + str(counter) + "_model/hp_input_deck.txt"
		print(path_fl_sav)
		fl_id         = open(path_fl_sav) #Read the hp file line by line
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

	#-----------------------------------------------;
	#  4. Sanity check on hp .txt files using glob  ;
	#-----------------------------------------------;
	hp_glob_txt_list = glob.glob(path + '1_InvDNNModel_ss_we/**/*.txt', recursive = True) #21875
	print(len(hp_glob_txt_list))

	#======================;
	# End processing time  ;
	#======================;
	toc = time.perf_counter()
	print('Time elapsed in seconds = ', toc - tic)