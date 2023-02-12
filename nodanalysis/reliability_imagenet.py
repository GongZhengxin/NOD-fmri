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

# change to path of current file
os.chdir(os.path.dirname(__file__))
# define path
dataset_root = '/nfs/z1/zhenlab/BrainImageNet'
fmriprep_path = f'{dataset_root}/NaturalObject/data/bold/derivatives/fmriprep'
ciftify_path = f'{dataset_root}/NaturalObject/data/bold/derivatives/ciftify'
nifti_path = f'{dataset_root}/NaturalObject/data/bold/nifti'

sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') and int(i[-2:])<10] )
result_path = './test_retest_reliability'
n_sub = len(sub_names)
n_class = 1000
num_ses, num_run, num_trial = 4, 10, 100 
vox_num = 59412
clean_code = 'hp128_s4'
support_path = './supportfiles'
result_path = f'./test_retest_reliability'

all_cross_ses_reliability = np.zeros((10, vox_num)) # 1-9: subjects, 10:mean across subjects
# for each subject
for sub_idx, sub_name in enumerate(sub_names):
    print(sub_name)
    label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
    # check whether label exists, if not then generate  
    if not os.path.exists(label_file):
        sub_events_path = pjoin(nifti_path, sub_name)
        df_img_name = []
        # find imagenet task
        imagenet_sess = [_ for _ in os.listdir(sub_events_path) if ('ImageNet' in _) and ('5' not in _)]
        imagenet_sess.sort()# Remember to sort list !!!
        # loop sess and run
        for sess in imagenet_sess:
            for run in np.linspace(1,10,10, dtype=int):
                # open ev file
                events_file = pjoin(sub_events_path, sess, 'func',
                                    '{:s}_{:s}_task-naturalvision_run-{:02d}_events.tsv'.format(sub_name, sess, run))
                tmp_df = pd.read_csv(events_file, sep="\t")
                df_img_name.append(tmp_df.loc[:, ['trial_type', 'stim_file']])
        df_img_name = pd.concat(df_img_name)
        df_img_name.columns = ['class_id', 'image_name']
        df_img_name.reset_index(drop=True, inplace=True)
        # add super class id
        superclass_mapping = pd.read_csv(pjoin(support_path, 'superClassMapping.csv'))
        superclass_id = superclass_mapping['superClassID'].to_numpy()
        class_id = (df_img_name.loc[:, 'class_id'].to_numpy()-1).astype(int)
        df_img_name = pd.concat([df_img_name, pd.DataFrame(superclass_id[class_id], columns=['superclass_id'])], axis=1)
        # make path
        if not os.path.exists(support_path):
            os.makedirs(support_path)
        df_img_name.to_csv(label_file, index=False)
        print(f'Finish preparing labels for {sub_name}')
    # load sub label file
    label_sub = pd.read_csv(label_file)['class_id'].to_numpy()
    label_sub = label_sub.reshape((num_ses, n_class))
    # define beta path
    beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
    if not os.path.exists(beta_sub_path):
        # extract from dscalar.nii
        results_fold = 'results'
        beta_sub = np.zeros(num_ses, num_run*num_trial, vox_num)
        for i_ses in range(num_ses):
            for i_run in range(num_run):
                run_name = f'ses-coco_task-imagnet{i_ses+1:02d}_run-{i_run+1}'
                beta_sub_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                beta_sub[i_ses, i_run*num_trial : (i_run + 1)*num_trial, :] = np.asarray(nib.load(beta_sub_path).get_fdata())
        # save session beta in ./supportfiles 
        np.save(beta_sub_path, beta_sub)
    else:
        beta_sub = np.load(beta_sub_path)
    # pick first session and standardization
    scaler = StandardScaler()
    # sort data 
    for sess_idx in range(num_ses):
        beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
    # compute the cross session reliability
    cross_ses_reliability = np.squeeze((np.dstack(tuple([np.corrcoef(beta_sub[:,:,ivox])[np.tril_indices(4, -1)] for ivox in range(vox_num)]))))
    all_cross_ses_reliability[sub_idx, :] = cross_ses_reliability.mean(axis=0)
all_cross_ses_reliability[-1,:] = np.mean(all_cross_ses_reliability[0:8,:], axis=0)
np.save(pjoin(result_path, f'/imagenet_{clean_code}_cross-ses-reliability.npy', all_cross_ses_reliability))
save_relibility = np.zeros((10, 91282))
save_relibility[:, 0:vox_num] = all_cross_ses_reliability
save_ciftifile(save_relibility, result_path, f'imagenet_{clean_code}_cross-ses-reliability.dtseries.nii')
print('##### save #####')

