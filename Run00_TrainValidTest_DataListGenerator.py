"""
Created on Tuesday, April 26th 2023
@author: sayimgokyar
Prepared by sayim gokyar to generate datalists for n-fold cross validation.
n_fold_crossvalidation value (default=5) should be provided by the user along with the data folder (default =./data) location.

For queries: sayim.gokyar@loni.usc.edu
"""
from __future__ import print_function, division
print(__doc__)

from os import listdir
import random

n_fold_crossvalidation = 5
data_path = './data/'

if __name__ == '__main__':

    DATA_FOLDERS = list([folder for folder in listdir(data_path) if folder.startswith('Sub')])
    random.shuffle(DATA_FOLDERS)        # Randomly shuffle the subject folders!
    
    nof_subjects = len(DATA_FOLDERS)    #This should return >10 for this work! (Since scans are going on, it is counting!)
    nof_test_folders = round(nof_subjects/n_fold_crossvalidation)  #Number of test folders for a single-fold (should be >=2 for 5-fold cross-validation)
    nof_valid_folders = nof_test_folders    #Number of validation folders for a single-fold! Number of folders for validation and test should be the same.
    nof_train_folders = nof_subjects - nof_valid_folders - nof_test_folders     # Remanining folders will be used for training for a given fold!
    
    # Generate Folder Lists for Train/Valid/Test for each Fold.
    for nn in range(1, 1+n_fold_crossvalidation):
        print('Fold:', nn)
        
        test_paths = DATA_FOLDERS[:nof_test_folders]    # Use the first n_test folders for testing
        valid_paths = DATA_FOLDERS[nof_test_folders:nof_test_folders+nof_valid_folders] # Use the next n_valid folders for validation
        train_paths = DATA_FOLDERS[nof_test_folders+nof_valid_folders:] # Use the remaining folders for training.
        
        print('Test Data:', test_paths, 'Validation Data:', valid_paths, 'Training Data:', train_paths)
        fname_train = data_path+'Folders_Fold'+str(nn)+'_Train.txt'
        with open(fname_train,'w') as ff:
            for item in train_paths:
                ff.write(item+"\n")
    	     
        fname_valid = data_path+'Folders_Fold'+str(nn)+'_Valid.txt'
        with open(fname_valid,'w') as ff:
          for item in valid_paths:
                 ff.write(item+"\n")
                 
        fname_test = data_path+'Folders_Fold'+str(nn)+'_Test.txt'
        with open(fname_test,'w') as ff:
          for item in test_paths:
                 ff.write(item+"\n")
        
        # Rotate the DATA_FOLDERS by number of test folders for each fold to avoid re-testing on the same data for different folds.
        DATA_FOLDERS.extend(DATA_FOLDERS[:nof_test_folders])
        del DATA_FOLDERS[:nof_test_folders]
    
    print('INFO: Data folders were assigned and lists generated at location: ', data_path)
    