import os
import numpy as np
import pandas as pd
import nibabel as nib
import random
import h5py
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
from os.path import join as pjoin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, zscore, ttest_1samp
from nod_utils import get_roi_data
from nltk.corpus import wordnet as wn
# define path
beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'
melodic_path= '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/melodic/'
fmriprep_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'
ciftify_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify'
nifti_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/nifti'

main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/'
network_path = '/nfs/z1/atlas/ColeAnticevicNetPartition/cortex_parcel_network_assignments.mat'
mask_path = '/nfs/z1/userhome/ZhouMing/workingdir/BIN/NaturalObject/data/code/nodanalysis/voxel_masks'

# time.sleep(7200)
##### new codes required by revised paper
##>> 
sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])])
n_sub = len(sub_names)
n_class = 1000
clean_code = 'clean_s4_poly3_mot'
beta_sum = np.zeros((n_sub, n_class, 59412))
for sub_idx, sub_name in enumerate(sub_names):
    label_sub = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
    if '07' in sub_name:
        label_sub = label_sub[0:2000]
    # define beta path
    beta_sub_path = pjoin(beta_path, sub_name, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
    beta_sub = np.load(beta_sub_path)
    # reshape
    n_sess = int(beta_sub.shape[0]/1000)
    label_sub = label_sub.reshape((n_sess, 1000))
    beta_sub = beta_sub.reshape(n_sess, 1000, 59412)
    # pick first session and standardization
    scaler = StandardScaler()
    # sort data 
    for sess_idx in range(n_sess):
        beta_sub[sess_idx] = beta_sub[sess_idx, np.argsort(label_sub[sess_idx])]
    # select the first session
    # beta_sub = scaler.fit_transform(beta_sub[0])
    # beta_sub = scaler.fit_transform(beta_sub.transpose())
    beta_sum[sub_idx] = beta_sub[0]
    print(f'{sub_idx}')
# np.save(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy', beta_sum)

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
# get visual areas
df_visual = pd.read_csv('HCP-MMP1_visual-cortex1.csv')
roi_names = list(df_visual.area_name[0:62])
visual_mask = get_roi_data(None, roi_names)
visual_area_indices = np.where(visual_mask==1)[0]

# load animacy annotations
animacy_arr = sio.loadmat('/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/exp/animate_or_not.mat')['animate_label']
sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])!=7])
n_sub = len(sub_names)
n_class = 1000
animacy_map = np.zeros((n_sub, 59412))
# load beta data
beta_sum = np.load(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy')

###>> t test
from scipy.stats import ttest_ind
sub_t_map = np.zeros((31,beta_sum.shape[-1]))
for i_sub in range(30):
    sub_beta = beta_sum[i_sub, :, :]
    animate_betas = sub_beta[np.where(animacy_arr==1)[1],:]
    inanimate_betas = sub_beta[np.where(animacy_arr==-1)[1],:]
    sub_t_map[i_sub, :] = ttest_ind(animate_betas, inanimate_betas)[0]
# sub_t_map[-1,:] = np.nanmean(sub_t_map[0:30,:], axis=0)
sub_t_map[-1,:] = ttest_1samp(sub_t_map[0:30,:], popmean=0)[0]
save_sub_t_map = np.zeros((31,91282))
save_sub_t_map[:,0:59412] = sub_t_map
animacy_map_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/inter-subject-animacymap'
save_ciftifile(save_sub_t_map, pjoin(animacy_map_path, f'all-sub-{clean_code}_t-wb-map.dtseries.nii'))
# 
save_sub_t_map = np.nan*np.zeros((31,91282))
save_sub_t_map[:,visual_area_indices] = sub_t_map[:, visual_area_indices]
save_ciftifile(save_sub_t_map, pjoin(animacy_map_path, f'all-sub-{clean_code}_t-visnet-map.dtseries.nii'))

# ###>> cohend computation
# beta_sum = beta_sum.transpose([0,2,1])
# # get weights of animacy contrast
# weights = np.array([v/490 if v==1 else v/510 for v in animacy_arr[0]])
# # calc cohend
# cohend = (beta_sum*weights).sum(axis=-1)/(beta_sum.std(axis=-1)+1e-10)
# # save the result in npy and dtseries nii
# animacy_map_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/inter-subject-animacymap'
# np.save(pjoin(animacy_map_path, 'all-sub_cohend-wb-map.npy'),cohend)
# animacy_cohend_map = np.nan*np.zeros((31,91282))
# animacy_cohend_map[0:30,0:59412] = cohend
# animacy_cohend_map[-1,0:59412] = cohend.mean(axis=0)
# save_ciftifile(animacy_cohend_map, pjoin(animacy_map_path, 'all-sub_cohend-wb-map.dtseries.nii'))
# #
# animacy_cohend_map = np.nan*np.zeros((31,91282))
# animacy_cohend_map[0:30,visual_area_indices] = cohend[:, visual_area_indices]
# animacy_cohend_map[-1,visual_area_indices] = cohend.mean(axis=0)[visual_area_indices]
# save_ciftifile(animacy_cohend_map, pjoin(animacy_map_path, 'all-sub_cohend-visnet-map.dtseries.nii'))

###>> iss calculation
# whole brain iss
calc_input = sub_t_map[0:30,:] # cohend
if np.isnan(calc_input).sum():
    del_column = np.where(np.isnan(calc_input)==1)[1]
    calc_input = np.delete(calc_input, del_column, axis=1)
wb_iss = np.corrcoef(calc_input)
pair_wb_iss = wb_iss[np.tril_indices(30,-1)]
np.save(pjoin(animacy_map_path, f'pair-{clean_code}_t_iss_wb.npy'), pair_wb_iss)
# visual area iss
# calc iss
calc_input = sub_t_map[0:30,visual_area_indices]
if np.isnan(calc_input).sum():
    del_column = np.where(np.isnan(calc_input)==1)[1]
    calc_input = np.delete(calc_input, del_column, axis=1)
visual_iss = np.corrcoef(calc_input)
pair_visual_iss = visual_iss[np.tril_indices(30,-1)]
np.save(pjoin(animacy_map_path, f'pair-{clean_code}_t_iss_visnet.npy'), pair_visual_iss)


# # plot iss histogram
# import matplotlib.pyplot as plt 
# # define plot utils
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams.update({'font.size': 10.5, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

# fig, ax = plt.subplots(1, 1, figsize=(10,6))
# ## 


# ax.legend()
# ax.tick_params("both", width=2.0, direction='in')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_linewidth(2.0)
# ax.spines['bottom'].set_linewidth(2.0)
# # ax.set_yticklabels([])
# ax.set_xticks([0,1,2,3,4,5,6,7])#
# layer_label = ['conv1','conv2','conv3','conv4','conv5','fc1','fc2','fc3']
# ax.set_xticklabels(layer_label,rotation=20)
# plt.show()

###################################################### old codes 
# # load select voxel info
# roimap = sio.loadmat(pjoin(main_path, 'MMP_mpmLR32k.mat'))['glasser_MMP']  # 1x59412
# network = sio.loadmat(network_path)['netassignments'] 
# network = [x[0] for x in network]
# select_network = [1, 2, 10, 11]
# visual_network_loc = np.array([True if network[int(x-1)] in select_network else False for x in roimap.squeeze()])
# visual_area_indices = np.where(visual_network_loc==1)[0]

# df_visual = pd.read_csv('HCP-MMP1_visual-cortex1.csv')
# roi_names = list(df_visual.area_name[0:62])
# # roi_names.extend(['TE1a','STSvp','STSva','STSdp','STSda','TGv','TGd','STGa','v23ab','d23ab'])
# visual_mask = get_roi_data(None, roi_names)
# visual_area_indices = np.where(visual_mask==1)[0]

# # # generate surface map
# def save_ciftifile(data, filename):
#     template = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify/sub-core02/MNINonLinear/Results/ses-ImageNet01_task-object_run-1/ses-ImageNet01_task-object_run-1_Atlas.dtseries.nii'
#     ex_cii = nib.load(template)
#     if len(data.shape) > 1:
#         ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
#     else:
#         ex_cii.header.get_index_map(0).number_of_series_points = 1
#         data = data[np.newaxis, :]
#     nib.save(nib.Cifti2Image(data.astype(np.float32), ex_cii.header), filename)

# # get p value mask
# def FDR_correct(ps,alpha):
#     ps = np.sort(np.squeeze(ps))
#     ref_arr = alpha*(np.arange(len(ps))+1)/len(ps)
#     k = np.where(ps > ref_arr)[0][0]-1
#     return k, ps[k]

# # data load
# animacy_arr = sio.loadmat('/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/exp/animate_or_not.mat')['animate_label']
# sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])])
# n_sub = len(sub_names)
# n_class = 1000
# animacy_map = np.zeros((n_sub, 59412))

# synset_file = '/nfs/z1/zhenlab/DNN/ImgDatabase/ImageNet_2012/label/synsets.txt'
# with open(synset_file,'r') as f:
#     words_pos_offset = [_.replace('\n', '') for _ in f.readlines()]
# get_synset = lambda x: wn.synset_from_pos_and_offset('n',x)
# synsets_list = [get_synset(int(_.replace('n',''))) for _ in words_pos_offset]
# human = wn.synset("human_being.n.01")
# # person = wn.synset("person.n.01")
# # living_thing = wn.synset("living_thing.n.01")human,
# animacy_continuum = np.array([[__.wup_similarity(_) for _ in synsets_list] for __ in [ human]])
# animacy_continuum = animacy_continuum.mean(axis=0)

# #
# from sklearn.linear_model import LinearRegression
# lr = LinearRegression(n_jobs=5)
# scaler = StandardScaler()
# beta_sum = np.load(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy')
# voxel_pattern = beta_sum[:,:,visual_area_indices]
# animacy_arr = np.atleast_2d(animacy_arr).transpose()
# animacy_beta_map = np.nan*np.zeros((n_sub, 91282))
# for sub_idx in range(voxel_pattern.shape[0]):
#     sub_voxel_pattern = voxel_pattern[sub_idx]
#     sub_voxel_pattern = scaler.fit_transform(sub_voxel_pattern)
#     lr.fit(animacy_arr, sub_voxel_pattern)
#     animacy_beta_map[sub_idx, visual_area_indices] = lr.coef_[:,0]
# from scipy.stats import ttest_1samp
# tmap, pmap = ttest_1samp(animacy_beta_map,popmean=0)
# animacy_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/inter-subject-animacymap/'
# save_ciftifile(tmap, pjoin(animacy_path, f'all-sub_raw_contrast_animacymap-t.dtseries.nii'))
# save_ciftifile(pmap, pjoin(animacy_path, f'all-sub_raw_contrast_animacymap-p.dtseries.nii'))

# ps = pmap[visual_area_indices]
# k,p_thres = FDR_correct(ps, 0.05) 
# from scipy.stats import t
# t_thres = t.ppf(1-p_thres/2, 30)

# for i in range(voxel_pattern.shape[0]):
#     voxel_pattern[i] = scaler.fit_transform(voxel_pattern[i])
#     voxel_pattern[i] = ((voxel_pattern[i].transpose() - voxel_pattern[i].mean(axis=1))/np.sqrt(np.square(voxel_pattern[i]).sum(axis=1))).transpose()
# mean_pattern = voxel_pattern.mean(axis=0)

# plt.figure(figsize=(4,4))
# plt.imshow(np.corrcoef(mean_pattern),vmin=-0.10,vmax=0.10,cmap='jet')
# plt.axis('off')
# plt.savefig('figures/RDMofvisual-voxstd.jpg',dpi=72*4,bbox_inches='tight')

# from sklearn.manifold import MDS
# embedding = MDS(n_components=2)
# # embedding = PCA(n_components=2)
# neural_manifold = embedding.fit_transform(mean_pattern)

# def rotate_2D(theta):
#     return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# rotate_manifold = np.dot(rotate_2D(90/180*np.pi),neural_manifold.transpose()).transpose()
# df_labels = pd.read_csv('/nfs/z1/userhome/GongZhengXin/NVP/Analysis_results/superClassMapping.csv')
# animate_cls = sorted(df_labels.iloc[np.where(np.squeeze(animacy_arr)==1)[0]].superClassID.unique())
# inanimate_cls = sorted(df_labels.iloc[np.where(np.squeeze(animacy_arr)==-1)[0]].superClassID.unique())
# # get the mean position of wach categories
# calc_index = rotate_manifold[:,0] #animacy_continuum 
# animate_dim1mean,inanimate_dim1mean = [], []
# for super_class in sorted(df_labels.superClassID.unique()):
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         animate_dim1mean.append(calc_index[indices].mean())
#     elif super_class in inanimate_cls:
#         inanimate_dim1mean.append(calc_index[indices].mean())
# # assign alpha values for each super class
# animate_alphas, inanimate_alphas = [], []
# animate_rank, inanimate_rank = np.argsort(animate_dim1mean), np.argsort(inanimate_dim1mean)[::-1]
# for i in range(len(animate_dim1mean)):
#     pos = np.where(animate_rank==i)[0]
#     animate_alphas.append(np.linspace(1,0.2,len(animate_dim1mean))[pos][0])
# for i in range(len(inanimate_dim1mean)):
#     pos = np.where(inanimate_rank==i)[0]
#     inanimate_alphas.append(np.linspace(1,0.2,len(inanimate_dim1mean))[pos][0])
# # plot
# _manifold =  rotate_manifold# neural_manifold
# fig,ax = plt.subplots(1,1,figsize=(4,4))
# for super_class in sorted(df_labels.superClassID.unique())[::-1]:#
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         aidx = int(np.where(animate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],marker='o',
#         label=class_label,color='#e30000',alpha=animate_alphas[aidx],s=18)#,edgecolors='red',linewidths=2
#     elif super_class in inanimate_cls:
#         aidx = int(np.where(inanimate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],marker='^',
#         label=class_label,color='#1400a3',alpha=inanimate_alphas[aidx],s=18)#,edgecolors='blue',linewidths=2
# # plt.legend(bbox_to_anchor=(1.05, 1))
# plt.axis('off')
# plt.savefig('figures/1000cate_visual-MDS_alpha-mdsdim1.jpg',bbox_inches='tight',dpi=72*4)
# plt.show()

# ###################################################################################################

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
#     # beta_sub = scaler.fit_transform(beta_sub[0])
#     # beta_sub = scaler.fit_transform(beta_sub.transpose())
#     # animacy_map[sub_idx] = [np.corrcoef(beta_sub[0][:,_], animacy_arr)[0,1] for _ in range(59412)]
#     animacy_map[sub_idx] = [np.corrcoef(beta_sub[0][:,_], animacy_continuum)[0,1] for _ in range(59412)]
#     print('Finish calc data:%s' % sub_name)

# animacy_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/inter-subject-animacymap/'
# # np.save(pjoin(animacy_path,f'all-sub_raw_scale-finegrid_animacymap.npy'), animacy_map)
# save_beta = np.zeros((int(n_sub),91282))
# save_beta[:,:59412] = animacy_map
# # save_ciftifile(save_beta, pjoin(animacy_path, f'all-sub_raw_scale-finegrid_animacymap.dtseries.nii'))


# scale = 'finegrid' #'coarse'
# finegrid_animacymap = pjoin(animacy_path, f'all-sub_raw_scale-{scale}_animacymap.npy')
# finegrid_animacymap = np.load(finegrid_animacymap) 

# toz = lambda x : 0.5*np.log((1+x)/(1-x))
# finegrid_animacymap = toz(finegrid_animacymap)

# tmap,pmap = np.nan*np.zeros((91282)),np.nan*np.zeros((91282))
# tmap[:59412],pmap[:59412] = ttest_1samp(finegrid_animacymap, popmean=0)

# save_ciftifile(tmap, pjoin(animacy_path, f'all-sub_raw_scale-{scale}_animacymap-t.dtseries.nii'))
# save_ciftifile(pmap, pjoin(animacy_path, f'all-sub_raw_scale-{scale}_animacymap-p.dtseries.nii'))

# # %%
# import nibabel as nib
# # get roi mask 
# VTC_OTC = ['V1','V2','V3','V4','V8', 'PH', 'TE2p', 'FFC', 'VVC', 'TE1a','STSvp','STSva','STSdp','STSda',
#             'VMV1', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3', 'PIT', 'TE2a', 'TF','TGv','TGd','MT','PeEc','TE1m','TE1p',
#             'LO1', 'LO2', 'LO3', 'PHT', 'MST', 'FST', 'V4t', 'TPOJ1', 'TPOJ2', 'TPOJ3','STGa']
# roi_mask = get_roi_data(None, VTC_OTC)
# roi_indices = np.where(roi_mask==1)[0]


# animacy_path = '/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/data/code/nodanalysis/inter-subject-animacymap/'
# scale = 'finegrid'
# pmap = nib.load(pjoin(animacy_path, f'all-sub_thres_scale-{scale}_animacymap-p.dtseries.nii')).get_fdata()
# tmap = nib.load(pjoin(animacy_path, f'all-sub_thres_scale-{scale}_animacymap-t.dtseries.nii')).get_fdata()
# pmap = pmap[0,:59412]
# k, p_thres = FDR_correct(pmap[roi_indices],0.05)
# sig_mask = np.zeros(91282)
# sig_mask[:59412] = (pmap <= p_thres)
# # voxel selection
# select_mask = np.logical_and(sig_mask, roi_mask)
# voxel_indices = np.where(select_mask==1)[0]

# # load subject data
# sub_names = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i[-2:])])
# n_sub = len(sub_names)
# n_class = 1000
# beta_sum = np.zeros((n_sub, n_class, 59412))
# for sub_idx, sub_name in enumerate(sub_names):
#     label_sub = pd.read_csv(pjoin(beta_path, sub_name, f'{sub_name}_imagenet-label.csv'))['class_id'].to_numpy()
#     if '07' in sub_name:
#         label_sub = label_sub[0:2000]
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
#     # beta_sub = scaler.fit_transform(beta_sub[0])
#     # beta_sub = scaler.fit_transform(beta_sub.transpose())
#     beta_sum[sub_idx] = beta_sub[0]
#     print(f'{sub_idx}')
# # np.save(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy', beta_sum)

# # arcoss subjct PCA validation
# n_components = 10
# pca = PCA(n_components)
# voxel_pattern = beta_sum[:,:,voxel_indices]
# for i in range(voxel_pattern.shape[0]):
#     voxel_pattern[i] = scaler.fit_transform(voxel_pattern[i])
#     voxel_pattern[i] = ((voxel_pattern[i].transpose() - voxel_pattern[i].mean(axis=1))/np.sqrt(np.square(voxel_pattern[i]).sum(axis=1))).transpose()
# scores = np.zeros((n_sub, n_class, n_components))
# expvar_ratio = np.zeros((n_sub, n_components))
# for sub_idx in range(n_sub):
#     target_pattern = voxel_pattern[sub_idx]
#     mean_pattern = voxel_pattern[np.delete(np.arange(n_sub), sub_idx)].mean(axis=0)
#     pca.fit(mean_pattern)
#     scores[sub_idx] = pca.transform(target_pattern)
#     expvar_ratio[sub_idx] = np.squeeze(pca.explained_variance_ratio_)
#     print(f'sub-{sub_idx}')
# mean_score = scores.mean(axis=0)


# # %% neural RDM
# beta_sum = np.load(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy')
# voxel_pattern = beta_sum[:,:,voxel_indices]
# print(voxel_indices.shape)
# for i in range(voxel_pattern.shape[0]):
#     voxel_pattern[i] = scaler.fit_transform(voxel_pattern[i])
#     voxel_pattern[i] = ((voxel_pattern[i].transpose() - voxel_pattern[i].mean(axis=1))/np.sqrt(np.square(voxel_pattern[i]).sum(axis=1))).transpose()
# mean_pattern = voxel_pattern.mean(axis=0)
# plt.figure(figsize=(2,2))
# plt.imshow(np.corrcoef(mean_pattern),vmin=-0.15,vmax=0.15,cmap='jet')
# plt.axis('off')
# plt.savefig('figures/RDMofotc-sig-voxstd.jpg',dpi=72*4,bbox_inches='tight')

# #%% generate RDM of wordnet and dichotonomy
# animacy_arr_2d = np.atleast_2d(animacy_arr)
# category_RDM = np.dot(animacy_arr_2d.transpose(),animacy_arr_2d)
# plt.figure(figsize=(2,2))
# plt.imshow(category_RDM,vmin=-0.15,vmax=0.15,cmap='jet')
# plt.axis('off')
# plt.savefig('figures/RDMofDichotomy.jpg',dpi=72*4,bbox_inches='tight')

# main_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/'
# wordnet = np.load(pjoin(main_path, 'rdm_wordnet_raw.npy'))
# hypernyms_raw = np.load(pjoin(main_path, 'hypernyms_idx.npy'), allow_pickle=True)
# hypernyms_idx = [i for item in hypernyms_raw for i in item]
# wordnet = wordnet[:,np.array(hypernyms_idx)]
# wordnet = wordnet[np.array(hypernyms_idx),:]
# plt.figure(figsize=(2,2))
# plt.imshow(wordnet,vmin=0.2,vmax=0.7,cmap='jet')
# plt.axis('off')
# plt.savefig('figures/RDMofWordnet.jpg',dpi=72*4,bbox_inches='tight')

# #%% partial cor & part cor (unique contribute)
# from sklearn.linear_model import LinearRegression
# wn_RDM = wordnet
# di_RDM = np.load('animacyRDM/cateRDM.npy')
# otc_RDM = np.load('animacyRDM/otcRDM-voxnorm.npy')
# frt_RDM = np.load('animacyRDM/frontalRDM-voxnorm.npy')

# low_triangle = np.where(np.tril(wn_RDM, k=-1)!=0)
# x_wn = wn_RDM[low_triangle]
# x_di = di_RDM[low_triangle]
# y_otc = otc_RDM[low_triangle]
# y_frt = frt_RDM[low_triangle]

# # simple correlation
# otc_wn_r = np.corrcoef(y_otc, x_wn)[0,1]
# otc_di_r = np.corrcoef(y_otc, x_di)[0,1]
# frt_wn_r = np.corrcoef(y_frt, x_wn)[0,1]
# frt_di_r = np.corrcoef(y_frt, x_di)[0,1]
# print(otc_wn_r,otc_di_r,frt_wn_r,frt_di_r)


# lr = LinearRegression(n_jobs=4)
# lr.fit(np.atleast_2d(x_wn).transpose(), x_di)
# residual_di = x_di - lr.predict(np.atleast_2d(x_wn).transpose())

# lr.fit(np.atleast_2d(x_di).transpose(), x_wn)
# residual_wn = x_wn - lr.predict(np.atleast_2d(x_di).transpose())
# # part correlation
# otc_wn_r = np.corrcoef(y_otc, residual_wn)[0,1]
# otc_di_r = np.corrcoef(y_otc, residual_di)[0,1]
# frt_wn_r = np.corrcoef(y_frt, residual_wn)[0,1]
# frt_di_r = np.corrcoef(y_frt, residual_di)[0,1]
# print(otc_wn_r,otc_di_r,frt_wn_r,frt_di_r)
# ###
# lr.fit(np.atleast_2d(x_wn).transpose(), y_otc)
# res_otc_for_di = y_otc - lr.predict(np.atleast_2d(x_wn).transpose())
# lr.fit(np.atleast_2d(x_di).transpose(), y_otc)
# res_otc_for_wn = y_otc - lr.predict(np.atleast_2d(x_di).transpose())

# lr.fit(np.atleast_2d(x_wn).transpose(), y_frt)
# res_frt_for_di = y_frt - lr.predict(np.atleast_2d(x_wn).transpose())
# lr.fit(np.atleast_2d(x_di).transpose(), y_frt)
# res_frt_for_wn = y_frt - lr.predict(np.atleast_2d(x_di).transpose())
# # partial correlation
# otc_wn_r = np.corrcoef(res_otc_for_wn, residual_wn)[0,1]
# otc_di_r = np.corrcoef(res_otc_for_di, residual_di)[0,1]
# frt_wn_r = np.corrcoef(res_frt_for_wn, residual_wn)[0,1]
# frt_di_r = np.corrcoef(res_frt_for_di, residual_di)[0,1]
# print(otc_wn_r,otc_di_r,frt_wn_r,frt_di_r)

# # %% KNN
# beta_sum = np.load(f'sub1-30_{clean_code}_ses1_imagenet-beta.npy')
# print(voxel_indices.shape)
# voxel_pattern = beta_sum[:,:,voxel_indices]
# voxel_tmap = tmap[0,voxel_indices]
# voxel_tmap = voxel_tmap/voxel_tmap.sum()

# # feature scaling 
# voxel_pattern = voxel_pattern*voxel_tmap
# # 暂时不做 feature scaling
# # 但是要考虑每个类脑图的相对pattern会比较合适(避免各类的方差差异太大)
# mode_norm = lambda x: ((x.transpose() - x.mean(axis=1))/np.sqrt(np.square(x).sum(axis=1))).transpose()
# std_norm =  lambda x: scaler.fit_transform(x.transpose()).transpose()
# normtype = 'mode'
# norm_metric = None
# exec(f'norm_metric = {normtype}_norm')
# for i in range(voxel_pattern.shape[0]):
#     # normlization
#     voxel_pattern[i] = scaler.fit_transform(voxel_pattern[i])
#     voxel_pattern[i] = norm_metric(voxel_pattern[i])
# for sub_idx in range(beta_sum.shape[0]):
#     target_pattern = voxel_pattern[sub_idx]
#     # mean the group pattern and then renorm the std between categories
#     mean_pattern = voxel_pattern[np.delete(np.arange(n_sub), sub_idx)].mean(axis=0)
#     mean_pattern = norm_metric(mean_pattern)
#     knn_r = np.corrcoef(mean_pattern, target_pattern)[0:1000,1000:2000]
#     idfy_mat = np.argsort(knn_r, axis=0)
#     top1 = np.sum(np.diag(idfy_mat) == 999)
#     top5 = np.sum(np.diag(idfy_mat) >= 995)
#     print(sub_idx, ' ', top1, ' ',top5)

# # 发现用不显著的体素做KNN与显著体素做KNN看不出显著区别，可能的原因就是一团乱
# # %% two-pathway
# ventral_part = ['V8', 'PeEc', 'FFC', 'VVC','VMV1', 'VMV2', 'VMV3', 'PHA1', 'PHA2', 'PHA3']
# dorsal_part = ['TE1p','PH','MT','PIT','LO1', 'LO2', 'LO3', 'PHT', 'MST', 'FST', 'V4t', 'TPOJ1', 'TPOJ2', 'TPOJ3']

# v_indices = np.where(np.logical_and(sig_mask, get_roi_data(None,ventral_part))==1)[0]
# d_indices = np.where(np.logical_and(sig_mask, get_roi_data(None,dorsal_part))==1)[0]

# ventral_pattern = beta_sum[:,:,v_indices]
# dorsal_pattern = beta_sum[:,:,d_indices]
# print(ventral_pattern.shape, dorsal_pattern.shape)

# mode_norm = lambda x: ((x.transpose() - x.mean(axis=1))/np.sqrt(np.square(x).sum(axis=1))).transpose()
# std_norm =  lambda x: scaler.fit_transform(x.transpose()).transpose()
# normtype = 'mode'
# norm_metric = None
# exec(f'norm_metric = {normtype}_norm')
# for i in range(ventral_pattern.shape[0]):
#     # normlization
#     ventral_pattern[i] = scaler.fit_transform(ventral_pattern[i])
#     ventral_pattern[i] = norm_metric(ventral_pattern[i])
# for i in range(dorsal_pattern.shape[0]):
#     # normlization
#     dorsal_pattern[i] = scaler.fit_transform(dorsal_pattern[i])
#     dorsal_pattern[i] = norm_metric(dorsal_pattern[i])

# ventral_pattern = ventral_pattern.mean(axis=0)
# dorsal_pattern = dorsal_pattern.mean(axis=0)

# v_RDM = np.corrcoef(ventral_pattern)
# d_RDM = np.corrcoef(dorsal_pattern)

# # # feature_scale
# # fsc_v_pattern = (beta_sum[:,:,v_indices]*tmap[0,v_indices]).mean(axis=0)
# # fsc_d_pattern = (beta_sum[:,:,d_indices]*tmap[0,d_indices]).mean(axis=0)


# def draw_RDM(rdm):
#     vmin = np.percentile(rdm, 5)
#     vmax = np.percentile(rdm, 95)
#     plt.imshow(rdm, vmin=vmin, vmax=vmax, cmap='jet')

# def cor_rdms(rdm1, rdm2):
#     if rdm1.shape == rdm2.shape:
#         _triangle = np.tril(np.ones_like(rdm1),k=-1)!=0
#         return np.corrcoef(rdm1[_triangle],rdm2[_triangle])
#     else:
#         raise AssertionError('shape not match')

# # %% neural MDS
# from sklearn.manifold import MDS
# voxel_pattern = beta_sum[:,:,voxel_indices]
# for i in range(voxel_pattern.shape[0]):
#     voxel_pattern[i] = scaler.fit_transform(voxel_pattern[i])
#     voxel_pattern[i] = ((voxel_pattern[i].transpose() - voxel_pattern[i].mean(axis=1))/np.sqrt(np.square(voxel_pattern[i]).sum(axis=1))).transpose()
# mean_pattern = voxel_pattern.mean(axis=0)
# embedding = MDS(n_components=2)
# # embedding = PCA(n_components=2)
# neural_manifold = embedding.fit_transform(mean_pattern)

# def rotate_2D(theta):
#     return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# rotate_manifold = np.dot(rotate_2D(np.pi/4),neural_manifold.transpose()).transpose()
# import pandas as pd
# df_labels = pd.read_csv('/nfs/z1/userhome/GongZhengXin/NVP/Analysis_results/superClassMapping.csv')
# animate_cls = sorted(df_labels.iloc[np.where(np.squeeze(animacy_arr)==1)[0]].superClassID.unique())
# inanimate_cls = sorted(df_labels.iloc[np.where(np.squeeze(animacy_arr)==-1)[0]].superClassID.unique())
# # get the mean position of wach categories
# calc_index = rotate_manifold[:,0] #animacy_continuum 
# animate_dim1mean,inanimate_dim1mean = [], []
# for super_class in sorted(df_labels.superClassID.unique()):
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         animate_dim1mean.append(calc_index[indices].mean())
#     elif super_class in inanimate_cls:
#         inanimate_dim1mean.append(calc_index[indices].mean())
# # assign alpha values for each super class
# animate_alphas, inanimate_alphas = [], []
# animate_rank, inanimate_rank = np.argsort(animate_dim1mean), np.argsort(inanimate_dim1mean)[::-1]
# for i in range(len(animate_dim1mean)):
#     pos = np.where(animate_rank==i)[0]
#     animate_alphas.append(np.linspace(1,0.2,len(animate_dim1mean))[pos][0])
# for i in range(len(inanimate_dim1mean)):
#     pos = np.where(inanimate_rank==i)[0]
#     inanimate_alphas.append(np.linspace(1,0.2,len(inanimate_dim1mean))[pos][0])
# # plot
# _manifold = rotate_manifold
# fig,ax = plt.subplots(1,1,figsize=(2,2))
# for super_class in sorted(df_labels.superClassID.unique())[::-1]:#
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         aidx = int(np.where(animate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],marker='o',
#         label=class_label,color='#e30000',alpha=animate_alphas[aidx],s=3)#,edgecolors='red',linewidths=2
#     elif super_class in inanimate_cls:
#         aidx = int(np.where(inanimate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],marker='^',
#         label=class_label,color='#1400a3',alpha=inanimate_alphas[aidx],s=3)#,edgecolors='blue',linewidths=2
# # plt.legend(bbox_to_anchor=(1.05, 1))
# plt.axis('off')
# # plt.savefig('figures/1000cate_MDS_alpha-mdsdim1.jpg',bbox_inches='tight',dpi=72*4)
# plt.show()



# _manifold = rotate_manifold.transpose()
# colors = [['#ffa69d','#ff564c','#ff3627','#ff1c08','#d93c2f','#db1800','#ff1f1c','#ff720e','#fc6500','#cd6700','#ff387f','#973d00','#ff2755','#ff5f7a','#ff1c42'],
# ['#624eff','#a29dff','#00bcc1','#008095','#00b1c1','#007ab1','#0089f1','#0062bf','#0840ff','#0036db','#1616ff','#84f2ff','#08fffb','#95a9ff','#9ad1ff','#b1eaff']]
# alpha = 0.9
# fig,ax = plt.subplots(1,1,figsize=(10,10))
# for super_class in sorted(df_labels.superClassID.unique()):#[::-1]
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         cidx1, cidx2 = 0, int(np.where(animate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],
#         label=class_label,color=colors[cidx1][cidx2],alpha=alpha,s=144)#,edgecolors='red',linewidths=2
#     elif super_class in inanimate_cls:
#         cidx1, cidx2 = 1, int(np.where(inanimate_cls==super_class)[0])
#         ax.scatter(_manifold[indices, 0],_manifold[indices, 1],
#         label=class_label,color=colors[cidx1][cidx2],alpha=alpha,s=144)#,edgecolors='blue',linewidths=2
# # plt.legend(bbox_to_anchor=(1.05, 1))
# plt.axis('off')
# # plt.savefig('figures/1000cate_MDS.jpg',bbox_inches='tight')
# plt.show()

# # 3D plots
# embedding = MDS(n_components=3)
# neural_manifold = embedding.fit_transform(mean_pattern)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for super_class in df_labels.superClassID.unique():
#     indices = np.where(df_labels.superClassID.values==super_class)[0]
#     class_label = df_labels.iloc[df_labels.superClassID.values==super_class].superClassName.unique()[0]
#     if super_class in animate_cls:
#         cidx1, cidx2 = 0, int(np.where(animate_cls==super_class)[0])
#         ax.scatter(neural_manifold[indices, 0],neural_manifold[indices, 1],neural_manifold[indices, 2],
#         label=class_label,color=colors[cidx1][cidx2],alpha=alpha)
#     elif super_class in inanimate_cls:
#         cidx1, cidx2 = 1, int(np.where(inanimate_cls==super_class)[0])
#         ax.scatter(neural_manifold[indices, 0],neural_manifold[indices, 1],neural_manifold[indices, 2],
#         label=class_label,color=colors[cidx1][cidx2],alpha=alpha)
# # plt.legend()
# ax.view_init(elev=45, azim=120)
# plt.show()




# animacy_arr = sio.loadmat('/nfs/z1/userhome/GongZhengXin/NVP/NaturalObject/exp/animate_or_not.mat')['animate_label']
# for animacy_lable in [1,-1]:
#     indices = np.where(np.squeeze(animacy_arr)==animacy_lable)[0]
#     plt.scatter(neural_manifold[indices, 0],neural_manifold[indices, 1],label=animacy_lable)
# plt.show()




# %%
