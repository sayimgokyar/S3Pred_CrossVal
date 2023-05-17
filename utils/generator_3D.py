
from __future__ import print_function, division

import numpy as np
from os import listdir
from random import shuffle
import h5py
            
def data_generator_sar(data_file_list, img_size, file_type, testing = False, shuffle_epoch=True):

    files = sorted(data_file_list)
    nbatches = len(files)
    
    x = np.zeros((1,)+(img_size))
    y = np.zeros((1,)+(img_size[0],img_size[1],img_size[2],1))
    
    # print(np.shape(x))
    # print(np.shape(y))
    
    while True:
        
        if shuffle_epoch:
            shuffle(files)
        else:
            files = sorted(data_file_list)
        
        for batch_cnt in range(nbatches):
            for file_cnt in range(1):

                file_ind = batch_cnt*1+file_cnt
                files = list(files)
                
                if file_type == 'B1P': # Single Channel B1P
                    in_path = '%s.B1P'%(files[file_ind][0:-4])  #B1P path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/30
                
                elif file_type == 'B1PR': # Single Channel B1PR
                    in_path = '%s.B1M'%(files[file_ind][0:-4])  #B1PR path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/30
                
                elif file_type == 'GRE': # Single Channel MRI
                    in_path = '%s.T1'%(files[file_ind][0:-4])  #MRI path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/np.nanmax(data)
                
                elif file_type == 'B1P_GRE': # Double Channel B1P and MRI
                    in_path = '%s.B1P'%(files[file_ind][0:-4])  #B1P path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/30
                    in_path = '%s.T1'%(files[file_ind][0:-4])  #MRI path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 1] = data/np.nanmax(data)
                
                elif file_type == 'B1PR_GRE': # Double Channel B1PR and MRI
                    in_path = '%s.B1M'%(files[file_ind][0:-4])  #B1PR path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/30
                    in_path = '%s.T1'%(files[file_ind][0:-4])  #MRI path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 1] = data/np.nanmax(data)
                
                elif file_type == 'B1P_B1PR': # Double Channel B1P and B1PR
                    in_path = '%s.B1P'%(files[file_ind][0:-4])  #B1P path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 0] = data/30
                    in_path = '%s.B1M'%(files[file_ind][0:-4])  #B1PR path
                    with h5py.File(in_path,'r') as f:
                        data = f['ds'] [:]
                        x[file_cnt,..., 1] = data/30
                        
                        
                elif file_type == 'B1P_B1PR_GRE':   # Three Channel
                     in_path = '%s.B1P'%(files[file_ind][0:-4])  #B1P path
                     with h5py.File(in_path,'r') as f:
                         data = f['ds'] [:]
                         x[file_cnt,..., 0] = data/30
                         #print('B1P Max/30: ', np.nanmax(x[file_cnt,..., 0]))
                     in_path = '%s.B1M'%(files[file_ind][0:-4])  #B1PR path
                     with h5py.File(in_path,'r') as f:
                         data = f['ds'] [:]
                         x[file_cnt,..., 1] = data/30
                         #print('B1PR Max/30: ', np.nanmax(x[file_cnt,..., 1]))
                     in_path = '%s.T1'%(files[file_ind][0:-4])  #MRI path
                     with h5py.File(in_path,'r') as f:
                         data = f['ds'] [:]
                         x[file_cnt,..., 2] = data/np.nanmax(data)
                         #print('MRI Max_normalized: ', np.nanmax(x[file_cnt,..., 2]))    
                         
                else:
                    print('ERROR from GENERATOR: Channel numbers and input types do not match!, Use B1P or T1 for single channel data. File Type is ignored for multichannel scenario')
    
                #print('INFO from GENERATOR: NOF_Channels:',img_size[3], 'INFO from GENERATOR: Input_Path', in_path)
                # SAR is the same for all cases! There is just one Ground Truth data.
                sar_path = '%s.SAR'%(files[file_ind][0:-4])
                with h5py.File(sar_path,'r') as f:
                    sar = f['ds'][:]
                    y[file_cnt,..., 0] = sar
                
                
                fname = files[file_ind]
                
            if testing is False:
                yield (x, y)
            else:
                yield (x, y, fname)
