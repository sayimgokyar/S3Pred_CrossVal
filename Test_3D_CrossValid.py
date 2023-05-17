"""
Created on April 28th 2023
@author: sayimgokyar
Prepared by sayim gokyar for testing 3D UNETs by using three-channel data. 

- Make sure you pointed model_weights in the main!

Set the file_types as:
%%%%%%%%%%%%%% Single Channel Models %%%%%%%%%%%%%%%%%%%%
- file_type == 'B1P': Test models trained on .B1p files,
- file_type == 'B1PR': Test models trained on .B1PR files,
- file_type == 'GRE' : Test models trained on .T1 files,

%%%%%%%%%%%%%% Double Channel Models %%%%%%%%%%%%%%%%%%%%
- file_type == 'B1P_B1PR': Test models trained on .B1P and .B1PR files,
- file_type == 'B1P_GRE': Test models trained on .B1P and .T1 files,
- file_type == 'B1PR_GRE': Test models trained on .B1PR and .T1 files,

%%%%%%%%%%%%%% Three Channel Model  %%%%%%%%%%%%%%%%%%%%%
- file_type == 'B1P_B1PR_GRE': Test models trained on .B1P, .B1PR, and .GRE files.

For queries: sayim.gokyar@loni.usc.edu
"""

from __future__ import print_function, division
print(__doc__)

from scipy import io
import numpy as np
import math
import tensorflow as tf

from image_similarity_measures.quality_metrics import rmse, psnr, fsim
import h5py
import time
from os import listdir
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse as nrmse

from tensorflow.keras import backend as K
from utils.generator_3D import data_generator_sar
from utils.models import unet_3d_model as Model

file_types = ('B1P', 'B1PR', 'GRE', 'B1P_B1PR', 'B1P_GRE', 'B1PR_GRE', 'B1P_B1PR_GRE')
input_data_folders = './data/'

lyrs=4 #number of stages for 3D UNET
conv_filters= 64
ModelName = Model.__name__

# Specify directories
test_result_path = './results'

SARh = []
SAR_GT_max = []
SAR_Pred_max = []
SAR3D_GT_max = []
SAR3D_Pred_max = []
MSE = []
NRMSE = []
SSIM = []
PSNR = []
RMSE = []
FSIM = []

INPUT_IMAGES = []
PREDICTIONS = []
GT = []
SummaryForPaperRevision = []

def test_sar(test_result_path, test_data_paths, file_type, CrossValNum, model_weights):
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
        
    img_cnt = 0

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    test_files = sorted([input_data_folders+folder+'/'+file for folder in test_data_paths for file in listdir(input_data_folders+folder) if file.endswith('.SAR')])
    
    print('INFO: Number of test files:', len(test_files))
    #test_files = sorted([f for f in listdir(test_path) if f.endswith('.SAR')])
    ntest = len(test_files)
    
    first_data = '%s'%(test_files[0])  #Read the first SAR data and return its size
    with h5py.File(first_data,'r') as f:
        data = f['ds'][:]  
    size = np.shape(data)
    #print('INFO: size of the data:', size)
         
    img_size = (size[0], size[1], size[2], nof_channels)
    
    # create the unet model
    model = Model(img_size, lyrs, conv_filters)
    model.load_weights(model_weights);

    start = time.time()
    
    # Iterature through the files to be tested 
    for x_test, y_test, fname in data_generator_sar(test_files, img_size, file_type, testing= True, shuffle_epoch=False):
        t0 = time.time()    #start timer for prediction duration
        
        recon = model.predict(x_test, batch_size = 1)
        t1 = time.time()    #stop timer for the prediction and print it!
        print('--'*5)
        print('Counter: %d, of %d' % (img_cnt+1, ntest))
        print('Duration for single prediction = %0.3f seconds.'  %  (t1-t0))
        
        #Zeroise every channel
        ch0 = 0*x_test[0,:,:,:,0]
        ch1 = ch0
        ch2 = ch0

        #Determine the type of inputs from the file_type
        if file_type == 'B1P': # Single Channel B1P
            ch2 = x_test[0,..., 0]
            ch2_colormap = 'viridis'
            titles = ('', '', file_type, 'GT', 'Pred.', 'Error') 
        elif file_type == 'B1PR': # Single Channel B1PR
            ch2 = x_test[0,..., 0]
            ch2_colormap = 'viridis'
            titles = ('', '', file_type, 'GT', 'Pred.', 'Error')
        elif file_type == 'GRE': # Single Channel MRI
            ch2 = x_test[0,..., 0]
            ch2_colormap = 'gray'
            titles = ('', '', 'MRI', 'GT', 'Pred.', 'Error')
               
        elif file_type == 'B1P_GRE': # Double Channel B1P and MRI
            ch1 = x_test[0,..., 0]
            ch2 = x_test[0,..., 1]
            ch2_colormap = 'gray'
            titles = ('', 'B1P', 'MRI', 'GT', 'Pred.', 'Error')
               
        elif file_type == 'B1PR_GRE': # Double Channel B1PR and MRI
            ch1 = x_test[0,..., 0]
            ch2 = x_test[0,..., 1]
            ch2_colormap = 'gray'
            titles = ('', 'B1PR', 'MRI', 'GT', 'Pred.', 'Error')
               
        elif file_type == 'B1P_B1PR': # Double Channel B1P and B1PR
            ch1 = x_test[0,..., 0]
            ch2 = x_test[0,..., 1]
            ch2_colormap = 'viridis'
            titles = ('', 'B1P', 'B1PR', 'GT', 'Pred.', 'Error')
                                             
        elif file_type == 'B1P_B1PR_GRE':   # Three Channel
            ch0 = x_test[0,..., 0]    
            ch1 = x_test[0,..., 1]
            ch2 = x_test[0,..., 2]
            ch2_colormap = 'gray'
            titles = ('B1P', 'B1PR', 'MRI', 'GT', 'Pred.', 'Error')    
                        
        else:
            print('ERROR from GENERATOR: Channel numbers and input types do not match!, Use B1P or T1 for single channel data. File Type is ignored for multichannel scenario')
    
            
        im_recon = recon[0,:,:,:,0]
        PREDICTIONS.append(im_recon)
        
        f2 = h5py.File(fname ,'r')
        im_gt = f2['ds'][:,:,:]
        f2.close()
        print('INFO: GT data has been read from:', fname[13:-4])
        #print('INFO: Length of fname:', len(fname[13:-4]))
        
        GT.append(im_gt)
        gt_max = np.nanmax(im_gt)
        gt_pred = np.nanmax(im_recon)
        
        SAR3D_GT_max.append(gt_max)
        SAR3D_Pred_max.append(gt_pred)
        
        print('3D_GT_max = %0.3f, 3D_Pred_max = %0.3f' % (np.nanmax(np.nanmax(im_gt)), np.nanmax(np.nanmax(im_recon))) )

        for kkk in range(22, 105, 1):
            
            ch0_image = ch0[:, :, kkk]
            ch1_image = ch1[:, :, kkk]
            ch2_image = ch2[:, :, kkk]
            
            IM_pred = im_recon[:, :, kkk]
            IM_GT = im_gt[:, :, kkk]
            
            mse = mean_squared_error(IM_GT, IM_pred)
            ss_ind = ssim(IM_GT, IM_pred, data_range=1)
            nrmse_ = nrmse(IM_GT, IM_pred, normalization='min-max')  
            p_snr_ = psnr(np.expand_dims(IM_GT, axis=2), np.expand_dims(IM_pred, axis=2))
            rmse_ = rmse(np.expand_dims(IM_GT, axis=2), np.expand_dims(IM_pred, axis=2))
            fsim_ = fsim(np.expand_dims(IM_GT, axis=2), np.expand_dims(IM_pred, axis=2))      
            
            #### Calculate the maximum corrdinate of the im_gt
            maxima = np.nanmax(IM_GT)
            maxima_pred = np.nanmax(IM_pred[0:127, 0:127])
            if maxima>=0.1:
                sarh = (-maxima+np.nanmax(IM_pred))/maxima 
                if kkk%20==0:
                    pass
                    #print('GT_max = %.3f, Pred_max = %.3f' %(maxima, maxima_pred))
            else:
                sarh = 0
            
            
            SAR_GT_max.append(maxima)
            SAR_Pred_max.append(maxima_pred)
            
            if math.isinf(mse) is False: MSE.append(mse)
            if math.isinf(nrmse_) is False: NRMSE.append(nrmse_)
            if math.isinf(p_snr_) is False: PSNR.append(p_snr_)
            if math.isinf(rmse_) is False: RMSE.append(rmse_)
            
            SSIM.append(ss_ind)
            SARh.append(sarh)
            FSIM.append(fsim_) 
 
            fig = plt.figure(num=None, figsize=[5, 2.0], dpi=320.0, frameon=False)
            fig.suptitle ('%s, %s, \n pSNR=%5.1f dB, SSIM=%.1f%%, FSIM=%.1f%% \n GT_max=%0.2f (W/kg)  Pred_max=%.2f (W/kg)' % (fname, kkk, p_snr_, 100*ss_ind, 100*fsim_, gt_max, gt_pred))

            ax0 = fig.add_subplot(1,6,1, aspect='equal') 
            im=ax0.imshow(np.flipud(ch0_image),  vmin=0, vmax=1.0, cmap='viridis')
            ax0.title.set_text(titles[0])
            plt.xticks([])
            plt.yticks([])

            ax0 = fig.add_subplot(1,6,2, aspect='equal') 
            im=ax0.imshow(np.flipud(ch1_image),  vmin=0, vmax=1.0, cmap='viridis')
            ax0.title.set_text(titles[1])
            plt.xticks([])
            plt.yticks([])

            ax0 = fig.add_subplot(1,6,3, aspect='equal') 
            im=ax0.imshow(np.flipud(ch2_image),  vmin=0, vmax=1.0, cmap=ch2_colormap)
            ax0.title.set_text(titles[2])
            plt.xticks([])
            plt.yticks([])
        
            ax1 = fig.add_subplot(1,6,4, aspect='equal')
            im=ax1.imshow(np.flipud(IM_GT),  vmin=0, vmax=1.0, cmap='viridis')
            ax1.title.set_text(titles[3])
            plt.xticks([])
            plt.yticks([])
            
            ax1 = fig.add_subplot(1,6,5, aspect='equal')
            im=ax1.imshow(np.flipud(IM_pred),  vmin=0, vmax=1.0, cmap='viridis')
            ax1.title.set_text(titles[4])
            plt.xticks([])
            plt.yticks([])
            
            ax1 = fig.add_subplot(1,6,6, aspect='equal')
            im=ax1.imshow(np.flipud(np.abs(IM_pred-IM_GT)),  vmin=0, vmax=1.0, cmap='viridis')
            ax1.title.set_text(titles[5])
            plt.xticks([])
            plt.yticks([])
            
            plt.subplots_adjust(top=0.55, bottom=0.03, left=0.03, hspace=0.02, wspace=0.02)
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="8%", pad=-0.05)
            plt.colorbar(im, cax=cax, orientation="vertical")
        
            plt.savefig(test_result_path + fname[13:-4] + '_'+str(nof_channels)+'Ch' +'_slice_' + '%03d'%kkk  + '.png')
            #plt.show()
            plt.close()
            
        img_cnt += 1
        if img_cnt == ntest:
            break

    Summary = [[100*np.nanmean(MSE), 100*np.nanstd(MSE)],
               [100*np.nanmean(RMSE), 100*np.nanstd(RMSE)],
               [100*np.nanmean(NRMSE), 100*np.nanstd(NRMSE)],
               [100*np.nanmean(SARh),100*np.nanstd(SARh)],
               [np.nanmean(PSNR), np.nanstd(PSNR)],
               [100*np.nanmean(SSIM),100*np.nanstd(SSIM)],
               [100*np.nanmean(FSIM),100*np.nanstd(FSIM)],
               ]
    SummaryForPaperRevision.append([100*np.nanmean(SSIM),100*np.nanstd(SSIM), 100*np.nanmean(FSIM), 100*np.nanstd(FSIM), 100*np.nanmean(MSE), 100*np.nanstd(MSE), np.nanmean(PSNR), np.nanstd(PSNR)])
    
    
    print('--'*50)
    print('Overall Summary for: %s' %fname)
    print(' MSE= %0.3f \u00B1 %0.3f'  %  (100*np.nanmean(MSE), 100*np.std(MSE))) 
    print(' RMSE= %0.3f \u00B1 %0.3f'  %  (100*np.nanmean(RMSE), 100*np.nanstd(RMSE)))
    print(' NRMSE= %0.3f \u00B1 %0.3f'  %  (100*np.nanmean(NRMSE), 100*np.nanstd(NRMSE)))
    print(' SARh= %0.1f \u00B1 %0.3f'  %  (100*np.nanmean(SARh),100*np.nanstd(SARh)))
    print(' pSNR= %0.2f \u00B1 %0.3f dB'  %  (np.nanmean(PSNR), np.nanstd(PSNR)))  
    print(' SSIM= %0.1f \u00B1 %0.3f'  %  (100*np.nanmean(SSIM),100*np.nanstd(SSIM)))
    print(' FSIM= %0.3f \u00B1 %0.3f'  %  (100*np.nanmean(FSIM),100*np.nanstd(FSIM)))

    #io.savemat('./results/Fold'+str(CrossValNum)+'_SummaryForPaperRevision.mat', {"SummaryForPaperRev":SummaryForPaperRevision})
    #io.savemat(test_result_path + '/'+'statistics' + '.mat', {"Summary":Summary, "SummaryForPaperRev":SummaryForPaperRevision, "MSE": MSE, "NRMSE": NRMSE, "SSIM":SSIM,  "SARh":SARh,
    #                                            'PSNR':PSNR, "RMSE":RMSE, "FSIM":FSIM, "SAR_GT_max":SAR_GT_max, "SAR_Pred_max":SAR_Pred_max, "SAR3D_GT_max":SAR3D_GT_max, "SAR3D_Pred_max":SAR3D_Pred_max})
    
    #io.savemat(test_result_path + '/'+'FullData' + '.mat', {"Summary":Summary, "GT": GT, "Input_Images": INPUT_IMAGES, "Predictions":PREDICTIONS, "SAR3D_GT_max":SAR3D_GT_max, "SAR3D_Pred_max":SAR3D_Pred_max})

    end = time.time()
    print(' Elapsed time = %0.1f seconds.'  %  (end-start)) 
    print('--'*50) 
    
if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use one GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print('--'*50)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            print(e)
    

    #test_sar(test_result_path, test_path, file_types, test_batch_size, model_weights)
    # Repeat the following for each fold of cross-validation
    for nn in range(1, 2):
        Fold = 'Fold'+str(nn)
        
        # Read The Testing Folders from file
        fname_test = input_data_folders+'Folders_Fold'+str(nn)+'_Test.txt'
        test_data_paths = []
        with open(fname_test, 'r') as fp:
            for line in fp:
                x = line[:-1]
                test_data_paths.append(x)
        
        print('INFO: Fold:', nn, '\nINFO: Test Data:', test_data_paths)
        
        # Repeat the following for each file-type inside each cross-validation step
        for file_type in file_types:
            result_path = './results/'+Fold+'/'+file_type+'/'
            model_weights = './weights/'+Fold+'_'+file_type+'.h5'

            print(model_weights)         
            print('INFO: File type:', file_type)
            print('INFO: Result path:', result_path)
            SARh = []
            SAR_GT_max = []
            SAR_Pred_max = []
            SAR3D_GT_max = []
            SAR3D_Pred_max = []
            MSE = []
            NRMSE = []
            SSIM = []
            PSNR = []
            RMSE = []
            FSIM = []

            INPUT_IMAGES = []
            PREDICTIONS = []
            GT = []
            
            SummaryForPaperRevision = []
            test_sar(result_path, test_data_paths, file_type, nn, model_weights)
    
    
    #io.savemat('./results/SummaryForPaperRevision.mat', {"SummaryForPaperRev":SummaryForPaperRevision})


















