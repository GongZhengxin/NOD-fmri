import os
import numpy as np
import pandas as pd
import scipy.io as sio
from os.path import join as pjoin
from sklearn.preprocessing import  StandardScaler
from scipy.stats import ttest_ind, ttest_1samp
from nod_utils import get_roi_data, save_ciftifile, prepare_imagenet_data

# change to path of current file
os.chdir(os.path.dirname(__file__))
# define path
dataset_root = 'PATHtoDataset'
fmriprep_path = f'{dataset_root}/derivatives/fmriprep'
ciftify_path = f'{dataset_root}/derivatives/ciftify'
nifti_path = f'{dataset_root}'
support_path = './supportfiles'
network_path = pjoin(support_path, 'cortex_parcel_network_assignments.mat') # visual netword rois

##>> load brain data
sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub')])
n_sub = len(sub_names)
n_class = 1000
clean_code = 'hp128_s4'
# first session of each participants are prepared before
ses1_beta_file =  pjoin(support_path, f'sub1-30_{clean_code}_ses1_imagenet-beta.npy')
if not os.path.exists(ses1_beta_file):
    beta_sum = np.zeros((n_sub, n_class, 59412))
    for sub_idx, sub_name in enumerate(sub_names):
        # define beta path
        beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
        if not os.path.exists(beta_sub_path):
            prepare_imagenet_data(dataset_root, sub_names=[sub_name])
        beta_sub = np.load(beta_sub_path)
        label_sub = pd.read_csv(pjoin(support_path, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
        # reshape
        n_sess = beta_sub.shape[0]
        label_sub = label_sub.reshape((n_sess, 1000))
        # pick first session and standardization
        scaler = StandardScaler()
        # sort data 
        for sess_idx in range(n_sess):
            beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
        # select the first session
        # beta_sub = scaler.fit_transform(beta_sub[0])
        # beta_sub = scaler.fit_transform(beta_sub.transpose())
        beta_sum[sub_idx] = beta_sub[0]
        print(f'load sub-{sub_idx+1}')
    np.save(ses1_beta_file, beta_sum)
else:
    beta_sum = np.load(ses1_beta_file)
# get visual areas
df_visual = pd.read_csv(pjoin(support_path, 'HCP-MMP1_visual-cortex1.csv'))
roi_names = list(df_visual.area_name[0:62])
visual_mask = get_roi_data(None, roi_names)
visual_area_indices = np.where(visual_mask==1)[0]

# load animacy annotations
animacy_arr = sio.loadmat(pjoin(support_path, 'animate_or_not.mat'))['animate_label']
sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub')])
n_sub = len(sub_names)
n_class = 1000
animacy_map = np.zeros((n_sub, 59412))

###>> t test
sub_t_map = np.zeros((int(n_sub+1),beta_sum.shape[-1]))
for i_sub in range(n_sub):
    sub_beta = beta_sum[i_sub, :, :]
    animate_betas = sub_beta[np.where(animacy_arr==1)[1],:]
    inanimate_betas = sub_beta[np.where(animacy_arr==-1)[1],:]
    sub_t_map[i_sub, :] = ttest_ind(animate_betas, inanimate_betas)[0]
# t test
sub_t_map[-1,:] = ttest_1samp(sub_t_map[0:n_sub,:], popmean=0)[0]
save_sub_t_map = np.zeros((int(1+n_sub),91282))
save_sub_t_map[:,0:59412] = sub_t_map
animacy_map_path = './inter-subject-animacymap'
# save_ciftifile(save_sub_t_map, pjoin(animacy_map_path, f'all-sub-{clean_code}_t-wb-map.dtseries.nii'))
save_sub_t_map = np.nan*np.zeros((int(1+n_sub),91282))
save_sub_t_map[:,visual_area_indices] = sub_t_map[:, visual_area_indices]
save_ciftifile(save_sub_t_map, pjoin(animacy_map_path, f'{n_sub}-sub-{clean_code}_t-visnet-map.dtseries.nii'))

###>> iss calculation
# # whole brain iss
# calc_input = sub_t_map[0:30,:]
# if np.isnan(calc_input).sum():
#     del_column = np.where(np.isnan(calc_input)==1)[1]
#     calc_input = np.delete(calc_input, del_column, axis=1)
# wb_iss = np.corrcoef(calc_input)
# pair_wb_iss = wb_iss[np.tril_indices(30,-1)]
# np.save(pjoin(animacy_map_path, f'pair-{clean_code}_t_iss_wb.npy'), pair_wb_iss)

# visual area iss
# calc iss
calc_input = sub_t_map[0:n_sub,visual_area_indices]
if np.isnan(calc_input).sum():
    del_column = np.where(np.isnan(calc_input)==1)[1]
    calc_input = np.delete(calc_input, del_column, axis=1)
visual_iss = np.corrcoef(calc_input)
pair_visual_iss = visual_iss[np.tril_indices(n_sub,-1)]
np.save(pjoin(animacy_map_path, f'{n_sub}subs-pair-{clean_code}_t_iss_visnet.npy'), pair_visual_iss)

