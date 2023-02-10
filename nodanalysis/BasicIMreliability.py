import os
import json
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore
from os.path import join as pjoin
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from nod_utils import save_ciftifile

# define path
beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'
# melodic_path= '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/melodic/'
# fmriprep_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'
# ciftify_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify'
# nifti_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/nifti'

sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])<=10 and int(i[-2:]) != 7] )
result_path = '/nfs/z1/userhome/GongZhengXin/NVP/Analysis_results/data_paper/result/ISCmap'
n_sub = len(sub_names)
n_class = 1000
vox_num = 59412
clean_code = 'clean_s4'

all_cross_ses_reliability = np.zeros((10, vox_num)) # 1-9: subjects, 10:mean across subjects
# for each subject 
for sub_idx, sub_name in enumerate(sub_names):
    print(sub_name)
    label_sub = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
    # define beta path
    beta_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
    beta_sub = np.load(beta_sub_path)
    # reshape
    n_sess = int(beta_sub.shape[0]/n_class)
    label_sub = label_sub.reshape((n_sess, n_class))
    beta_sub = beta_sub.reshape(n_sess, n_class, vox_num)
    # pick first session and standardization
    scaler = StandardScaler()
    # sort data 
    for sess_idx in range(n_sess):
        beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
    # compute the cross session reliability
    cross_ses_reliability = np.squeeze((np.dstack(tuple([np.corrcoef(beta_sub[:,:,ivox])[np.tril_indices(4, -1)] for ivox in range(vox_num)]))))
    all_cross_ses_reliability[sub_idx, :] = cross_ses_reliability.mean(axis=0)
all_cross_ses_reliability[-1,:] = np.mean(all_cross_ses_reliability[0:8,:], axis=0)
np.save(f'/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/test_retest_reliability/IM_{clean_code}_cross-ses-reliability.npy', all_cross_ses_reliability)
save_relibility = np.zeros((10, 91282))
save_relibility[:, 0:vox_num] = all_cross_ses_reliability
save_ciftifile(save_relibility, f'/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/test_retest_reliability/IM_{clean_code}_cross-ses-reliability.dtseries.nii')
print('##### save #####')

