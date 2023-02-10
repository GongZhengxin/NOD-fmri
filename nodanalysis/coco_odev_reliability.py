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

# define path
beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'
melodic_path= '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/melodic/'
fmriprep_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'
ciftify_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify'
nifti_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/nifti'

from scipy.stats import pearsonr

# # generate surface map
def save_ciftifile(data, filename):
    template = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/sub-core02/MNINonLinear/Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'
    ex_cii = nib.load(template)
    if len(data.shape) > 1:
        ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    else:
        ex_cii.header.get_index_map(0).number_of_series_points = 1
        data = data[np.newaxis, :]
    nib.save(nib.Cifti2Image(data.astype(np.float32), ex_cii.header), filename)
# %% averaged subjects inter-subject reliability
# sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])])
# result_path = '/nfs/z1/userhome/GongZhengXin/NVP/Analysis_results/data_paper/result/ISCmap'
# n_sub = len(sub_names)
# n_class = 1000
# beta_sum = np.zeros((n_sub, n_class, 59412))
# for sub_idx, sub_name in enumerate(sub_names):
#     label_sub = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
#     # define beta path
#     beta_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_imagenet-beta_clean_ridge.npy')
#     beta_sub = np.load(beta_sub_path)
#     # reshape
#     n_sess = int(beta_sub.shape[0]/1000)
#     label_sub = label_sub.reshape((n_sess, 1000))
#     beta_sub = beta_sub.reshape(n_sess, 1000, 59412)
#     # pick first session and standardization
#     scaler = StandardScaler()
#     # sort data 
#     for sess_idx in range(n_sess):
#         beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
#     # select the first session
#     beta_sub = beta_sub[0]
#     beta_sum[sub_idx] = zscore(beta_sub, axis=None)#scaler.fit_transform(beta_sub)
#     print('Finish load data:%s'%sub_name)

# file = '30-odd-even-20repeats_imagenet-sessionnormed_isc.dtseries.nii'
# isc_path = pjoin(result_path, f'{file}')
# n_repeats = 20
# isc_map = np.zeros((n_repeats, 59412))
# if not os.path.exists(isc_path):
#     np.random.seed(2022)
#     for repeat in range(n_repeats):
#         print(f'start repeat {repeat+1}')
#         idx1 = np.random.choice(30,15,replace=False)
#         idx2 = np.array(list(set(np.arange(30)) - set(idx1)))
#         # odd_idx = np.linspace(0,30,15,endpoint=False).astype(np.int8)
#         # even_idx = np.linspace(1,29,15,endpoint=True).astype(np.int8)

#         odd_subjects_mean = beta_sum[idx1].mean(axis=0)
#         even_subjects_mean = beta_sum[idx2].mean(axis=0)

#         isc_map[repeat,:] = [pearsonr(odd_subjects_mean[:,_], even_subjects_mean[:,_])[0] for _ in range(59412)]
#         # save isc map
#     isc_map = isc_map.mean(axis=0)
#     isc_map_save = np.zeros((91282))
#     isc_map_save[:59412] = isc_map
#     save_ciftifile(isc_map_save, pjoin(result_path, f'{file}'))
# else:
#     isc_map = np.array(nib.load(isc_path).get_fdata().tolist()).squeeze()[:59412]

# %% generally ISC computing 
# # Load Imagenet beta for 30 subjects 
# sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])])
# result_path = '/nfs/z1/userhome/GongZhengXin/NVP/Analysis_results/data_paper/result/ISCmap'
# n_sub = len(sub_names)
# n_class = 1000
# beta_sum = np.zeros((n_sub, n_class, 59412))
# for sub_idx, sub_name in enumerate(sub_names):
#     label_sub = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
#     # define beta path
#     beta_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_imagenet-beta_clean_ridge.npy')
#     beta_sub = np.load(beta_sub_path)
#     # reshape
#     n_sess = int(beta_sub.shape[0]/1000)
#     label_sub = label_sub.reshape((n_sess, 1000))
#     beta_sub = beta_sub.reshape(n_sess, 1000, 59412)
#     # pick first session and standardization
#     scaler = StandardScaler()
#     # sort data 
#     for sess_idx in range(n_sess):
#         beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
#     # average acorss session
#     beta_sub = beta_sub.mean(axis=0)
#     beta_sum[sub_idx] = scaler.fit_transform(beta_sub)
#     print('Finish load data:%s'%sub_name)

# isc_path = pjoin(result_path, f'30mean_imagenet_isc.dtseries.nii')
# if not os.path.exists(isc_path):
#     isc_map = np.zeros((n_sub, 59412))
#     for voxel_idx in range(beta_sum.shape[-1]):
#         voxel_pattern = beta_sum[:, :, voxel_idx]
#         # ISC was computed by correlation of each per participant with the mean pattern of remaining n-1 participants
#         for sub_idx in range(n_sub):
#             target_pattern = voxel_pattern[sub_idx]
#             mean_pattern = voxel_pattern[np.delete(np.arange(n_sub), sub_idx)].mean(axis=0)
#             isc_map[sub_idx, voxel_idx] = pearsonr(target_pattern, mean_pattern)[0]
#         print('Finish voxel:%05d'%voxel_idx)
#     isc_map = isc_map.mean(axis=0)
#     # save isc map
#     isc_map_save = np.zeros((91282))
#     isc_map_save[:59412] = isc_map
#     save_ciftifile(isc_map_save, pjoin(result_path, f'30mean_imagenetisc.dtseries.nii'))
# else:
#     isc_map = np.array(nib.load(isc_path).get_fdata().tolist()).squeeze()[:59412]

# %% COCO odd-even intra-subject reliability
from scipy.stats import zscore
# Load COCO beta for 10 subjects 
sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])<=10])
result_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/test_retest_reliability'
n_sub = len(sub_names)
n_class = 120
clean_code = 'clean_s4'
beta_sum = np.zeros((n_sub, 2), dtype=np.object)
for sub_idx, sub_name in enumerate(sub_names):
    # define beta path
    beta_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_coco-beta_{clean_code}_ridge.npy')
    beta_sub = np.load(beta_sub_path)
    # beta_sub = zscore(beta_sub, axis=(1,2))
    beta_sum[sub_idx,0] = zscore(beta_sub[[0,2,4,6,8]].mean(axis=0), axis=0)
    beta_sum[sub_idx,1] = zscore(beta_sub[[1,3,5,7,9]].mean(axis=0), axis=0)
    print('Finish load data:%s'%sub_name)

isc_path = pjoin(result_path, f'coco-zscore-{clean_code}_withinsub_reliability.dtseries.nii')
if not os.path.exists(isc_path):
    isc_map = np.zeros((n_sub, 59412))
    for sub_idx in range(n_sub):
        odd_mean = beta_sum[sub_idx,0]
        eve_mean = beta_sum[sub_idx,1]
        isc_map[sub_idx, :] = np.array([pearsonr(odd_mean[:,_], eve_mean[:,_])[0] for _ in range(59412)])
        print(f'finish sub {sub_idx}')
    isc_map = isc_map.mean(axis=0)
    # save isc map
    isc_map_save = np.zeros((91282))
    isc_map_save[:59412] = isc_map
    save_ciftifile(isc_map_save, pjoin(result_path, f'coco-zscore-{clean_code}_withinsub_reliability.dtseries.nii'))
else:
    isc_map = np.array(nib.load(isc_path).get_fdata().tolist()).squeeze()[:59412]
