# S3Pred_CrossVal
Cross Validation codes and data for the Subject Specific SAR Prediction Study. 

Definitions added after the submitted manuscript is accepted for publication in Magnetic Resonance in Medicine (with submission ID: MRM-23-23728.R1), entitled as "Deep learning-based local SAR prediction using B1 maps and structural magnetic resonance images of the head for parallel transmission at 7T"

1- Data should be requested from the Corresponding Author of the publication (i.e., ). A Material Transfer Agreement of USC should be signed between the USC and asking parties to share the data.

2- Once you obtained the data folders use (or modify) the following files:

2.1. Run00_TrainValidTest_DataListGenerator.py: This scans the number of subject folders and creates an n-fold cross validation file lists for training/validation and testing. Default setting is five. You may modify the code for your needs or use your own code.

2.2. Once the train/valid/test splits created in "./data/" directory, you can use the "/Train_3D_CrossValid.py" file for n-fold cross validation of 7 different network architectures. Modify the code accordingly! 

2.3. After trainings complete, you can run "Test_3D_CrossValid.py" to generate predictions. Weights folder should include the network weights along with the loss curves (.csv or .xlsx files). Prediction results will be saved to ./results/.../ folders in .png format depending on the type of the network used. For each folder, there will be two matlab files containing performance metrics (SSIM, FSIM, RMSE, etc.) and prediction maps along with ground truths.

3- Once you generated results folders, more specifically .png files, you can use your own routines to generate .mp4 video files. 

4- ./results/Github_Exp2Sim_ValidationResults.pdf: contains the comparison of experimental and simulated B1+ maps for each scan.

5- ./results/ResultsMatFiles4GitHub/: This directory contains the results in .mat form along with the .m files to generate plots obtained in this work.

6- ./results/Paper01_Figures_Tables_Rev02_withFolds.pdf: This document contains the plots generated by using the data and .m files provided in the above folder.

7- ./results/UploadMedia: This folder contains the .mp4 files mentioned in the original article.

8- ./utils: This folder contains the data_generator and model files. You may use/write your own files depending on the network that you wanna utilize.
