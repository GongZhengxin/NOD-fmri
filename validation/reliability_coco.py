import os
import json
import subprocess
import numpy as np
import pandas as pd
import pickle as pkl
import nibabel as nib
from scipy.stats import zscore, pearsonr
from os.path import join as pjoin
from sklearn.preprocessing import StandardScaler
from nod_utils import save_ciftifile

# change to path of current file
os.chdir(os.path.dirname(__file__))
# define path
dataset_root = 'PATHtoDataset'
fmriprep_path = f'{dataset_root}/derivatives/fmriprep'
ciftify_path = f'{dataset_root}/derivatives/ciftify'
nifti_path = f'{dataset_root}'
# >> COCO odd-even intra-subject reliability
# result path
result_path = f'./test_retest_reliability'
if not os.path.exists(result_path):
    os.makedirs(result_path)
# Load COCO beta for 10 subjects 
sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') and int(i[-2:])<=9])
n_sub = len(sub_names)
num_run = 10
n_class = 120
clean_code = 'hp128_s4'
# initial average beta
beta_sum = np.zeros((n_sub, 2), dtype=np.object)
for sub_idx, sub_name in enumerate(sub_names):
    sub_data_path = f'./supportfiles/{sub_name}_coco-beta_{clean_code}_ridge.npy'
    if not os.path.exists(sub_data_path):
        # extract from dscalar.nii
        results_fold = 'results'
        run_names = [ f'ses-coco_task-coco_run-{_+1}' for _ in range(num_run)]
        sub_beta = np.zeros((num_run, n_class, 59412))
        for run_idx, run_name in enumerate(run_names):
            beta_sub_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
            sub_beta[run_idx, :, :] = np.asarray(nib.load(beta_sub_path).get_fdata())
        # save session beta in ./supportfiles 
        np.save(sub_data_path, sub_beta)
    else: 
        sub_beta = np.load(sub_data_path)
    beta_sum[sub_idx,0] = zscore(sub_beta[[0,2,4,6,8]].mean(axis=0), axis=0)
    beta_sum[sub_idx,1] = zscore(sub_beta[[1,3,5,7,9]].mean(axis=0), axis=0)
    print('Finish load data:%s'%sub_name)
# reliability result file
result_file = f'coco-zscore-{clean_code}_withinsub_reliability.dtseries.nii'
reliability_path = pjoin(result_path, result_file)
if not os.path.exists(reliability_path):
    reliability_map = np.zeros((n_sub, 59412))
    for sub_idx in range(n_sub):
        odd_mean = beta_sum[sub_idx,0]
        eve_mean = beta_sum[sub_idx,1]
        reliability_map[sub_idx, :] = np.array([pearsonr(odd_mean[:,_], eve_mean[:,_])[0] for _ in range(59412)])
        print(f'finish sub {sub_idx+1}')
    reliability_map = reliability_map.mean(axis=0)
    # save reliability map
    reliability_map_save = np.zeros((91282))
    reliability_map_save[:59412] = reliability_map
    save_ciftifile(reliability_map_save, pjoin(result_path, result_file))
else:
    reliability_map = np.array(nib.load(reliability_path).get_fdata().tolist()).squeeze()[:59412]
