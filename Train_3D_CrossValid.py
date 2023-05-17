"""
Created on Tuesday, Feb 28th 10:56:37 2023
@author: sayimgokyar
Prepared by sayim gokyar in Feb 2023 for training 3D UNETs by using either single channel data (with extensions .B1P, B1PR, or .T1) 
or three-channel data. User is supposed enter the file type only to run the code.
1- file_type == 'B1P': Train models by using .B1p files only,
2- file_type == 'GRE' : Train models by using .T1 files only,
3- file_type == 'B1P_GRE': Train models by using .B1P and .T1 files.
4- file_type == 'B1P_B1PR_GRE': Train models by using three channels (.B1P, .B1PR, .GRE)

Here
B1P: B1P of the scans
B1PR: Phase-reversed B1P of the scans
GRE:  MRI of the corresponding scans
SAR: Ground truth for the corresponding scans.

For queries: sayim.gokyar@loni.usc.edu
"""

from __future__ import print_function, division
print(__doc__)
import os
from os import listdir
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.generator_3D import  data_generator_sar
from utils.models import unet_3d_model as Model

ModelName = Model.__name__
FILE_TYPES = ('B1P', 'B1PR', 'GRE', 'B1P_B1PR', 'B1P_GRE', 'B1PR_GRE')

initial_lrate = 1e-4
lyrs=4                          #number of stages in 3D UNET
conv_filters=64                # Number of features at the initial stage
n_epochs = 200
n_fold_crossvalidation = 5      # Number of Cross-Validations

model_weights = None    #Initial weights are None for default. you can load pretrained weights if you want!
cp_save_path = './weights/'
input_data_folders = './data/'

def MAE(y_true, y_pred):        #Mean absolute error
    return K.mean(K.abs(y_true - y_pred), axis=None)
def wMAE(y_true, y_pred):       #weighted-MAE
    return K.mean(tf.math.multiply(K.abs(y_true - y_pred), y_true/K.mean(y_true)), axis=None)
def wMSE(y_true, y_pred):       #weighted-MSE
    return K.mean(tf.math.multiply(K.square(y_true - y_pred), y_true/K.mean(y_true)), axis=None)

def my_loss_fn(y_true, y_pred):   #Different weights can be assigned to the components of loss function
    MAE = K.mean(K.abs(y_true - y_pred), axis=None)
    wMAE = K.mean(tf.math.multiply(K.abs(y_true - y_pred), y_true/K.mean(y_true)), axis=None)
    wMSE = K.mean(tf.math.multiply(K.square(y_true - y_pred), y_true/K.mean(y_true)), axis=None) 
    my_loss_fn = MAE + wMAE + wMSE
    if tf.math.reduce_max(y_pred)< tf.math.reduce_max(y_true): # You may wanna add underestimation penalty to here!
        return my_loss_fn
    else:
        return my_loss_fn


def train_sar(train_paths, valid_paths, cp_save_path, n_epochs, n_fold, file_type):
    # Determine the number of channels from the file_type
    if file_type=='B1P' or file_type=='B1PR' or file_type =='GRE':
        nof_channels = 1        
    elif file_type =='B1P_B1PR' or file_type=='B1P_GRE' or file_type=='B1PR_GRE':
        nof_channels = 2
    elif file_type =='B1P_B1PR_GRE':
        nof_channels = 3
    else:
        nof_channels = 0
        print('Please correct file_type and try again!')
                
    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files = sorted([input_data_folders+folder+'/'+file for folder in train_paths for file in listdir(input_data_folders+folder) if file.endswith('.SAR')])
    valid_files = sorted([input_data_folders+folder+'/'+file for folder in valid_paths for file in listdir(input_data_folders+folder) if file.endswith('.SAR')])
    #print('INFO: First 5 train files:',train_files[0:5], '\nINFO: First 5 validation files:', valid_files[0:5])
            
    #Read the first data and return its size
    with h5py.File(train_files[0],'r') as f:
        data = f['ds'][:]  
    size = np.shape(data)         
    img_size = (size[0], size[1], size[2], nof_channels)
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_lrate, decay_steps=len(train_files)*5, decay_rate=0.95, staircase=True)
    
    # create the 3D UNET model
    model = Model(img_size, lyrs, conv_filters)
    #model.summary()

    if model_weights is not None:
        if os.path.isfile(model_weights):
            model.load_weights(model_weights)
            print('Model Weights loaded from: ', model_weights)

    # Set up the optimizer
    optimizer=Adam(learning_rate=lr_schedule, beta_1=0.90, beta_2=0.999)
    model.compile(optimizer=optimizer, loss=my_loss_fn, metrics=([MAE, wMAE, wMSE]))
    
    # Set callbacks per epoch
    cp_save_tag = 'Fold'+str(n_fold)+'_'+file_type
    cp_callback   = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '.h5', monitor="val_loss", mode="min", save_best_only=True)
    #cp_callback   = ModelCheckpoint(cp_save_path + '/' + cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5', save_best_only=False)
    csv_callback = tf.keras.callbacks.CSVLogger('./weights/' + cp_save_tag + '.csv', append=True) 
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.01, patience=10, mode="auto", baseline=None, restore_best_weights=True)
    callbacks_list = [cp_callback, csv_callback]

    # Start the training    
    model.fit_generator(
            data_generator_sar(train_files, img_size, file_type),
            len(train_files),
            epochs=n_epochs,
            verbose=1,      #0: Silent, #1: Default-progress bar, #2: Single line per epoch
            validation_data=data_generator_sar(valid_files, img_size, file_type),
            validation_steps=1,
            callbacks=callbacks_list, 
            shuffle=False)
    

if __name__ == '__main__':
    #Set the GPU(s)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use one GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')	#Select appropriate GPU
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
    
    
    # Repeat the following for each fold of cross-validation
    for nn in range(1, 6):
        
        # Read The Training Folders from file
        fname_train = input_data_folders+'Folders_Fold'+str(nn)+'_Train.txt'
        train_paths = []
        with open(fname_train, 'r') as fp:
            for line in fp:
                x = line[:-1]
                train_paths.append(x)
        
        # Read The Validation Folders from file
        fname_valid = input_data_folders+'Folders_Fold'+str(nn)+'_Valid.txt'
        valid_paths = []
        with open(fname_valid, 'r') as fp:
            for line in fp:
                x = line[:-1]
                valid_paths.append(x)
                

        print('INFO: Fold:', nn, '\nINFO: Validation Data:', valid_paths, '\nINFO: Training Data:',  train_paths)

        # Repeat the following for each file-type inside each cross-validation step
        for file_type in FILE_TYPES:
            print('INFO: File type:', file_type)
            #model_weights = './weights/Fold'+str(nn)+'_'+file_type+'.h5' # Use the previous weights to continue later!
            #print('Model Weights is:', model_weights)
            train_sar(train_paths, valid_paths, cp_save_path, n_epochs, nn, file_type)












