import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from convnet_feat import get_features

# Functions for loading SHJ abstract data and images

# Filenames for SGJ stimuli and labels in abstract form
fn_shj_abstract = 'data/shj_stimuli.txt'
fn_shj_labels = 'data/shj_labels.txt'

def get_label_coding(loss_type):
	# Set coding for class A and class B
	POSITIVE = 1.
	if loss_type == 'hinge':
		NEGATIVE = -1.
	elif loss_type == 'll':
		NEGATIVE = 0.
	else:
		assert False
	return POSITIVE,NEGATIVE

def load_shj_abstract(loss_type):
	# Loads SHJ data from text file
	# 
	# Input
	#   loss_type : either ll or hinge loss
	#
	# Output
	#   X : [ne x dim tensor] stimuli as rows
	#   y_list : list of [ne tensor] labels, with a list element for each shj type
	stimuli = pd.read_csv(fn_shj_abstract, header=None).to_numpy()
	labels = pd.read_csv(fn_shj_labels, header=None).to_numpy()
	stimuli = stimuli.astype(float)	
	ntype = labels.shape[0]
	POSITIVE,NEGATIVE = get_label_coding(loss_type)
	labels_float = np.zeros(labels.shape,dtype=float)
	labels_float[labels == 'A'] = POSITIVE
	labels_float[labels == 'B'] = NEGATIVE
	X = torch.tensor(stimuli).float()	
	y_list = []
	for mytype in range(ntype):
		y = labels_float[mytype,:].flatten()
		y = torch.tensor(y).float()
		y_list.append(y)
	return X,y_list

def process_shj_images():
	# Return
	#  X : [ne x dim tensor] stimuli as rows
	print(" Passing SHJ images through ConvNet...")
	stimuli,images = get_features('data','resnet18')
	print(" Done.")
	stimuli = stimuli.data.numpy().astype(float)
	X = torch.tensor(stimuli).float()
	return X

def load_shj_images(loss_type):
	# Loads SHJ data from images
	# 
	# Input
	#   loss_type : either ll or hinge loss
	#
	# Output
	#   X : [ne x dim tensor] stimuli as rows
	#   y_list : list of [ne tensor] labels, with a list element for each shj type
	X =  process_shj_images()
	X_abstract,y_list = load_shj_abstract(loss_type)	
	return X,y_list