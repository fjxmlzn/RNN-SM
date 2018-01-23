#!/usr/bin/env python
#-*-coding:utf-8 -*-

'''
Implementation of our RNN-SM algorithm
-------------
Based on paper:
    RNN-SM: Fast Steganalysis of VoIP Streams Using Recurrent Neural Network
-------------
Author: Zinan Lin
Email:  zinanl@andrew.cmu.edu
'''

import numpy as np
import os, pickle, random, datetime

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM

FOLDERS = [
	{"class" : 1, "folder" : "/data1/linzn/data/ch_g729a_100_10000ms_FEAT"}, # The folder that contains positive data files.
	{"class" : 0, "folder" : "/data1/linzn/data/ch_g729a_0_10000ms_FEAT"} # The folder that contains negative data files.
]

SAMPLE_LENGTH = 10000   # The sample length (ms)
BATCH_SIZE = 32         # batch size
ITER = 30               # number of iteration
FOLD = 5                # = NUM_SAMPLE / number of testing samples


'''
Get the paths of all files in the folder
'''
def get_file_list(folder):
	file_list = []
	for file in os.listdir(folder):
		file_list.append(os.path.join(folder, file))
	return file_list


'''
Read codeword file
-------------
input
    file_path
        The path to an ASCII file.
        Each line contains three integers: x1 x2 x3, which are the three codewords of the frame.
        There are (number of frame) lines in total. 
output
    the list of codewords
'''	
def parse_sample(file_path):
	file = open(file_path, 'r')
	lines = file.readlines()
	sample =[]
	for line in lines:
		line_split = line.strip("\r\n\t").strip().split("\t")
		sample.append(line_split)
	return sample


'''
Save variable in pickle
'''
def save_variable(file_name, variable):
	file_object = open(file_name, "wb")
	pickle.dump(variable, file_object)
	file_object.close()
	
    
'''
SS-QCCN training and testing
'''
if __name__ == '__main__':
	all_files = [(item, folder["class"]) for folder in FOLDERS for item in get_file_list(folder["folder"])]
	
	random.shuffle(all_files)
	
	save_variable('all_files.pkl', all_files)
	
	all_samples_x = [(parse_sample(item[0])) for item in all_files]
	all_samples_y = [item[1] for item in all_files]
	
	np_all_samples_x = np.asarray(all_samples_x)
	np_all_samples_y = np.asarray(all_samples_y)
	
	save_variable('np_all_samples_x.pkl', np_all_samples_x)
	save_variable('np_all_samples_y.pkl', np_all_samples_y)
	
	file_num = len(all_files)
	sub_file_num = int(file_num / FOLD)
	
	x_test = np_all_samples_x[0 : sub_file_num] # The samples for testing
	y_test = np_all_samples_y[0 : sub_file_num] # The label of the samples for testing
	
	x_train = np_all_samples_x[sub_file_num : file_num] # The samples for training
	y_train = np_all_samples_y[sub_file_num : file_num] # The label of the samples for training
	
	print("Building model")
	model = Sequential()
	model.add(LSTM(50, input_length = int(SAMPLE_LENGTH / 10), input_dim = 3, return_sequences = True)) # first layer 
	model.add(LSTM(50)) # second layer
	model.add(Dense(1)) # output layer
	model.add(Activation('sigmoid')) # activation function
	
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ["accuracy"])
    
	print("Training")
	for i in range(ITER):
		model.fit(x_train, y_train, batch_size = BATCH_SIZE, nb_epoch = 1, validation_data = (x_test, y_test))
		model.save('model_%d.h5' % (i + 1))