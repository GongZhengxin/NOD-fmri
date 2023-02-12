import os
import time
import numpy as np
import scipy.io as sio
import json
import warnings
from scipy.stats import zscore
from os.path import join as pjoin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nod_utils import prepare_imagenet_data, prepare_coco_data
from nod_utils import prepare_train_data, prepare_AlexNet_feature, get_voxel_roi
from nod_utils import relu, get_roi_data, ReceptiveFieldProcessor

warnings.simplefilter('ignore')
os.chdir(os.path.dirname(__file__))
# define path
dataset_root = '/nfs/z1/zhenlab/BrainImageNet'
ciftify_path = f'{dataset_root}/NaturalObject/data/bold/derivatives/ciftify'
# prepare data
prepare_imagenet_data(dataset_root)
prepare_coco_data(dataset_root)
# set subject
subs = ['sub-04']
clean_code = 'hp128_s4'

# subject models
for sub in subs:
    print(sub, clean_code)
    #--------------------
    # Brain part 
    #--------------------
    # get brain respones & nomorlized by run already
    brain_resp = prepare_train_data(sub,code=clean_code)

    # get test brain response
    brain_path = f'./supportfiles'
    test_file_name = f'{sub}_coco-beta_{clean_code}_ridge.npy'
    test_brain_resp = np.load(pjoin(brain_path, test_file_name))
    test_brain_resp = zscore(test_brain_resp, axis=(1,2))
    test_mean_pattern = test_brain_resp.mean(axis=0)
    test_mean_pattern = zscore(test_mean_pattern, None)

    # define rois
    vtc_areas = ['V8', 'PIT', 'FFC', 'VVC', ['VMV1', 'VMV2', 'VMV3'], ['LO1', 'LO2', 'LO3']]
    evc_areas = ['V1', 'V2', 'V3','V4']
    selected_rois = [__  for _ in [evc_areas, vtc_areas] for __ in _]

    # voxel_indices = np.where(np.sum([get_roi_data(None,_) for _ in selected_rois], axis=0)==1)[0]
    #--------------------
    # feature part 
    #--------------------
    imgnet_feature = prepare_AlexNet_feature(sub)
    coco_feature = prepare_AlexNet_feature('coco')
    
    #--------------------
    # model part 
    #--------------------
    t0 = time.time()
    # population receptive field
    retino_path = f'{ciftify_path}/{sub}/results/ses-prf_task-prf'
    file_name = f'{sub}_retinotopy-allrun-s4_params.mat' # raw-retinotopy but s4 data
    retino_mat = sio.loadmat(pjoin(retino_path, file_name))['result']
    n_vertices = brain_resp.shape[-1]
    retinotopy_params = np.zeros((n_vertices, 3))
    retinotopy_params[:,0] = retino_mat[0,0]['ang'][0:n_vertices,0]
    retinotopy_params[:,1] = retino_mat[0,0]['ecc'][0:n_vertices,0]*16/200
    retinotopy_params[:,2] = retino_mat[0,0]['rfsize'][0:n_vertices,0]*16/200
    # 
    # r2 = retino_mat[0,0]['R2'][np.where(np.isnan(retino_mat[0,0]['R2'])==0)]
    r2 = retino_mat[0,0]['R2'][0:59412, 0]
    for i, roi in enumerate(selected_rois) :
        roi_indices = np.where(get_roi_data(None, roi)==1)
        roi_r2 = r2[roi_indices]
        if np.isnan(roi_r2).sum():
            print(f'{roi} with value NaN')
            roi_r2[np.where(np.isnan(roi_r2)==1)] = 0
        top50 = np.argsort(roi_r2)[-50::]
        if i == 0:
            voxel_indices = roi_indices[0][top50]
        else:
            voxel_indices = np.hstack((voxel_indices, roi_indices[0][top50]))

    # we need to transfer the params into (x,y,size) model
    trans_retinotopy_params = np.zeros_like(retinotopy_params)
    trans_retinotopy_params[:,0] = np.cos(retinotopy_params[:,0]/180*np.pi)*retinotopy_params[:,1]
    trans_retinotopy_params[:,1] = np.sin(retinotopy_params[:,0]/180*np.pi)*retinotopy_params[:,1]
    trans_retinotopy_params[:,2] = retinotopy_params[:,2]
    # 
    log_file = f'./encoder_records/{sub}-{clean_code}_dnn-based_prfmodel_fe-relu-top50_lassocv_record.log'
    if not os.path.exists(log_file):
        lasso_log = open(log_file, 'w+')
        lasso_log.write(f'#,voxel,alpha,nfeatures,roi,layer,lasso-test-cor,lasso-test-r2,lr-test-cor,lr-test-r2,tc\n')
    else:
        lasso_log = open(log_file, 'a+')
    coefs = {}
    layers = ['conv1','conv2','conv3','conv4','conv5','fc1','fc2','fc3'] #
    n_job = 4
    num_cv = 4

    for iv, voxel in enumerate(voxel_indices):#samples
        spatial_maps = [ReceptiveFieldProcessor(16, *trans_retinotopy_params[voxel,:]).get_spatial_kernel(_) for _ in [55,27,13,13,13]]
        t1 = time.time()
        vox_roi = get_voxel_roi(voxel)
        if 'VMV' in vox_roi or 'LO' in vox_roi:
            vox_roi = vox_roi[0:-1]
        coefs[f'{voxel}'] = {}
        for i,layer in enumerate(layers):
            # different layer types have different loading metric
            if 'conv' in layer:
                weights = (spatial_maps[i] >= 0.5*np.max(spatial_maps[i])) * spatial_maps[i]
                weights = weights/(weights.sum())
                train_features = np.sum(relu(imgnet_feature.get(layer))*weights, axis=(2,3))
                test_features = np.sum(relu(coco_feature.get(layer))*weights, axis=(2,3))
            elif 'fc' in layer:
                train_features = np.squeeze(relu(imgnet_feature.get(layer)))
                test_features = np.squeeze(relu(coco_feature.get(layer)))
            # baseline linear model estimation 
            lr = make_pipeline(StandardScaler(), LinearRegression(n_jobs=n_job))
            lr.fit(train_features, brain_resp[:,voxel])
            # first try lassocv on a coarse range
            for k in range(2):
                test_r2  = []
                test_cor = []
                # initialize
                if k == 0:
                    if 'fc' in layer:
                        alphas = 10**np.linspace(-2,-1.2,20)
                    elif 'conv' in layer:
                        alphas = 10**np.linspace(-3,-1,30)
                # print(f'vox {voxel}, fit {k}, alpha:{alphas}')
                lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, n_jobs=n_job,cv=4,verbose=False, precompute=False))
                if 'fc' in layer:
                    lasso.fit(train_features, brain_resp[:,voxel])
                elif 'conv' in layer:
                    lasso.fit(train_features, brain_resp[:,voxel])
                non_zero_coefs = np.sum(lasso['lassocv'].coef_ != 0)
                best_alpha_pos = np.where(alphas == lasso['lassocv'].alpha_)[0]
                # check if the estimation is valid. Note that because of the regularization and non-linearity of fitting 
                # there might results into null model in which the coeffients are all compressed into 0, we exclude such result 
                if non_zero_coefs == 0 or best_alpha_pos == 0:
                    alphas = 10**np.linspace(np.round(np.log(alphas[0])/np.log(10))-2*(k+1),np.round(np.log(alphas[-1])/np.log(10))-2*(k+1),20)
                else:
                    # then narrrow down the range to fit again
                    if 'fc' in layer:
                        if lasso.score(train_features, brain_resp[:,voxel]) > 0:
                            break
                        else:
                            if best_alpha_pos == (len(alphas)-1):
                                alphas = 10**np.linspace(np.round(np.log(alphas[0])/np.log(10))+0.2*(k+1),np.round(np.log(alphas[-1])/np.log(10))+0.2*(k+1),20)
                            else:
                                alphas = np.linspace(alphas[best_alpha_pos-1],alphas[best_alpha_pos+2],20)
                            if np.diff(alphas).mean() < 0.005:
                                break
                    elif 'conv' in layer:
                        if lasso.score(train_features, brain_resp[:,voxel]) > 0:
                            break
                        else:
                            if best_alpha_pos == (len(alphas)-1):
                                alphas = 10**np.linspace(np.round(np.log(alphas[0])/np.log(10))+0.2*(k+1),np.round(np.log(alphas[-1])/np.log(10))+0.2*(k+1),20)
                            else:
                                alphas = np.linspace(alphas[best_alpha_pos-1],alphas[best_alpha_pos+2],20)
                            if np.diff(alphas).mean() < 0.005:
                                break
            non_zero_coefs = np.sum(lasso['lassocv'].coef_ != 0)
            # check model validity
            if non_zero_coefs == 0:
                random_fit = 1
            else:
                random_fit = 0
            # if invalid model then fit in a larger alpha range 
            while random_fit:
                # print(f'vox{voxel}.',end='')
                if non_zero_coefs == 0:
                    print('.',end='')
                    alphas = 10**np.linspace(-5,-3,30) # 0.05*np.random.rand(20)
                    lasso = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, n_jobs=n_job,cv=4,verbose=False, precompute=False))
                    if 'fc' in layer:
                        lasso.fit(train_features, brain_resp[:,voxel])
                    elif 'conv' in layer:
                        lasso.fit(train_features, brain_resp[:,voxel])
                    non_zero_coefs = np.sum(lasso['lassocv'].coef_ != 0)
                    random_fit = random_fit - 1
                else:
                    break
            # if random fit can not provide a valid model then return to linear model
            if np.sum(lasso['lassocv'].coef_ != 0) == 0:
                lasso = make_pipeline(StandardScaler(), LassoCV(alphas=[0], n_jobs=n_job,cv=4,verbose=False, precompute=False))
                lasso.fit(train_features, brain_resp[:,voxel])
                non_zero_coefs = np.sum(lasso['lassocv'].coef_ != 0)
            # estimate the accuracy 
            if 'fc' in layer: 
                lasso_r2 = lasso.score(test_features, test_mean_pattern[:,voxel])
                lasso_r = np.corrcoef(lasso.predict(test_features), test_mean_pattern[:,voxel])[0,1]
                lr_r2 = lr.score(test_features, test_mean_pattern[:,voxel])
                lr_r = np.corrcoef(lr.predict(test_features), test_mean_pattern[:,voxel])[0,1]
            elif 'conv' in layer:
                lasso_r2 = lasso.score(test_features, test_mean_pattern[:,voxel])
                lasso_r = np.corrcoef(lasso.predict(test_features), test_mean_pattern[:,voxel])[0,1]
                lr_r2 = lr.score(test_features, test_mean_pattern[:,voxel])
                lr_r = np.corrcoef(lr.predict(test_features), test_mean_pattern[:,voxel])[0,1]
            # 
            alpha = lasso['lassocv'].alpha_
            tc = time.time() - t1
            print(f'{iv}-vox{voxel},{layer},a={alpha},r={lasso_r}, tc{(time.time() - t1)}')
            coefs[f'{voxel}'][layer] = (np.where(lasso['lassocv'].coef_!=0)[0].tolist(), lasso['lassocv'].coef_[np.where(lasso['lassocv'].coef_!=0)[0]].tolist())
            lasso_log.write(f'{iv},{voxel},{alpha},{non_zero_coefs},{vox_roi},{layer},{lasso_r},{lasso_r2},{lr_r},{lr_r2},{tc}\n')
    lasso_log.close()
    coef_recod = open(f'./encoder_records/{sub}-{clean_code}_dnn-based_prfmodel_fe-relu-top50_lassocv_coef.json','w+')
    json.dump(coefs,coef_recod)
    coef_recod.close()
print(f'consume {time.time()-t0}')
