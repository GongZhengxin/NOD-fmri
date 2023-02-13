import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from nilearn.glm.first_level import make_first_level_design_matrix
from surf_utils import save2cifti, _poly_drift, _orthogonalize

# define path
dataset_path = '/nfs/z1/zhenlab/BrainImageNet'
ciftify_path = f'{dataset_path}/NaturalObject/data/bold/derivatives/ciftify'
nifti_path = f'{dataset_path}/NaturalObject/'
fprep_path = f'{dataset_path}/NaturalObject/data/bold/derivatives/fmriprep'
result_path = '/nfs/z1/zhenlab/BrainImageNet/Analysis_results/data_paper/result'
frac_path = pjoin(result_path, 'frac') 

# prepare params
alpha_params = [eval('1e%d'%idx) for idx in np.linspace(-5, 5, 11, dtype=int)]
sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub')])
target = 'imagenet' # imagenet or coco
clean_code = 'hp128_s4'
glm_mode = 'separate' # seperate or full GLM model in coco beta estimate
motion_on = True

# load hyperparamsinfo: shape: len(sub_names), len(frac_params), 59412
rdm_corr = np.load(pjoin(result_path, 'rdm_corr.npy')) 
alpha = alpha_params[rdm_corr.mean(axis=0).argmax()]

for sub_idx, sub_name in enumerate(sub_names):
    # prepare basic path
    beta_clean_path = pjoin(ciftify_path, sub_name, 'results', f'{sub_name}_{target}-beta_{clean_code}_ridge.npy')
    sub_tmp_path = pjoin(nifti_path, sub_name)
    sub_fpr_path = pjoin(fprep_path, sub_name)
    # if not os.path.exists(beta_clean_path):
    if target == 'imagenet':
        # prepare params
        tr, begin_dur, n_tr, n_event = 2, 16, 256, 100
        frame_times = np.arange(n_tr*10) * tr 
        trial_type = ['image%04d'%(idx+1) for idx in range(n_event * 10)]
        alpha_params = [eval('1e%d'%idx) for idx in np.linspace(-5, 5, 11, dtype=int)]
        # define beta path
        sub_tmp_path = pjoin(nifti_path, sub_name)
        _result_path = pjoin(ciftify_path, sub_name, 'MNINonLinear/Results/')
        sess_names = sorted([i for i in os.listdir(sub_tmp_path) if ('ImageNet' in i) and (int(i[-2:])<=4)])
        beta_clean = np.zeros((len(sess_names)*1000, 59412), dtype=np.float32)
        # loop in one subject
        for sess_idx, sess_name in enumerate(sess_names):
            sub_func_path = pjoin(sub_tmp_path, sess_name, 'func')
            events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i and 'rest' not in i and \
                                int(i.split('-')[-1].split('_')[0])<=10 and 'discard' not in i])
            dtseries_sess = np.zeros((2560, 91282))
            onset_sess = np.zeros((1000))
            duration_sess = np.zeros((1000))
            for run_idx, file in enumerate(events_file): 
                # define run name
                run_split = file.split('_')
                run_name = '_'.join(run_split[1:3]) + '_run-' + str(int(run_split[3].split('-')[-1]))
                # replace task name in sub02 and sub03
                if sub_name in ['sub-02', 'sub-03']:
                    run_name = run_name.replace('naturalvision', 'object')
                # fit design matrix based on trial onset time
                events = pd.read_csv(pjoin(sub_func_path, file), sep='\t')
                duration = events['duration']
                onset = events['onset'].to_numpy() + begin_dur
                # load time series
                dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                print(f'load {dtseries_path}')
                # concantenate all info into sess params
                dtseries_sess[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
                onset_sess[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
                duration_sess[n_event*run_idx:n_event*(run_idx+1)] = duration
            #
            sub_cnfd_path =pjoin(sub_fpr_path, sess_name, 'func')
            cnfd_file = sorted([i for i in os.listdir(sub_cnfd_path) if 'confounds_timeseries.tsv' in i and 'rest' not in i and \
                                int(i.split('-')[-2].split('_')[0])<=10 and 'discard' not in i])
            cnfd_sess = np.zeros((2560, 6))
            for run_idx, file in enumerate(cnfd_file):
                confounds = pd.read_csv(pjoin(sub_cnfd_path, file), sep='\t')
                cnfd_sess[n_tr*run_idx:n_tr*(run_idx+1),:] = confounds.loc[:,['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].values
            # prepare design matrix
            events = pd.DataFrame({'trial_type':trial_type, 'onset':onset_sess, 'duration':duration_sess})
            design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
            # design_matrix = make_first_level_design_matrix(frame_times, events, drift_model='polynomial', hrf_model='spm', drift_order=3)
            design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
            # add column for run effect
            run_effect = np.zeros((n_tr * 10, 10))
            for run_idx in range(10):
                run_effect[n_tr*run_idx:n_tr*(run_idx+1), run_idx] = 1
            run_effect = pd.DataFrame(run_effect, columns=['run-%02d'%(i+1) for i in range(10)])
            design_matrix = pd.concat([design_matrix.reset_index(drop=True), run_effect], 
                                    ignore_index=True, axis=1)
            # add poly drift
            order = 3
            run_drift = np.zeros((n_tr*10, order)) 
            run_frames = np.arange(n_tr) * tr
            drift_matrix = _poly_drift(order=order, frame_times=run_frames) 
            for i in range(order):
                run_drift[:,i] = np.tile(drift_matrix[:,i],10)
            run_drift = pd.DataFrame(run_drift, columns=['drift-%02d'%(i+1) for i in range(order)])
            design_matrix = pd.concat([design_matrix.reset_index(drop=True), run_drift], 
                                    ignore_index=True, axis=1)
            # add motion 
            sess_motion = _orthogonalize(cnfd_sess)
            sess_motion = pd.DataFrame(sess_motion, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
            design_matrix = pd.concat([design_matrix.reset_index(drop=True), sess_motion], 
                                    ignore_index=True, axis=1)
            # perform GLM
            reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_sess[:, :59412])
            beta = reg.coef_[:, :1000].transpose(1,0).astype(np.float32)
            beta_clean[1000*sess_idx:1000*(sess_idx+1)] = beta
            print('Finish performing GLM in %s %s'%(sub_name, sess_name))
    elif target == 'coco':
        # define params:
        tr, begin_dur, n_tr, n_event = 2, 16, 241, 120
        sess_name = 'ses-COCO'
        _result_path = pjoin(ciftify_path, sub_name, 'MNINonLinear/Results/')
        sub_func_path = pjoin(sub_tmp_path, sess_name, 'func')
        events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i and 'rest' not in i and 'discard' not in i])
        n_run = len(events_file)

        if glm_mode == 'full':
            #####
            # GLM in a full model
            frame_times = np.arange(n_tr*n_run) * tr 
            beta_clean = np.zeros((120, 59412), dtype=np.float32)
            # loop in one subject
            dtseries_sess = np.zeros((n_tr*n_run, 91282))
            onset_sess = np.zeros((n_event*n_run))
            duration_sess = np.repeat(0.5, (n_event*n_run))
            trial_type_sess = []
            for run_idx, file in enumerate(events_file): 
                # define run name
                run_split = file.split('_')
                run_name = '_'.join(run_split[1:3]) + '_run-' + str(int(run_split[3].split('-')[-1]))
                # fit design matrix based on trial onset time
                events_raw = pd.read_csv(pjoin(sub_func_path, file), sep='\t')
                onset = events_raw['onset'].to_numpy() + begin_dur
                label_sub = events_raw['trial_type'].to_numpy()
                trial_type = ['image%03d'%idx for idx in label_sub]
                # load time series
                dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                print(f'load {dtseries_path}')
                # concantenate all info into sess params
                dtseries_sess[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
                onset_sess[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
                trial_type_sess.extend(trial_type)
            # prepare design matrix
            events = pd.DataFrame({'trial_type':trial_type_sess, 'onset':onset_sess, 'duration':duration_sess})
            design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
            design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
            # add column for run effect
            run_effect = np.zeros((n_tr * n_run, n_run))
            for run_idx in range(n_run):
                run_effect[n_tr*run_idx:n_tr*(run_idx+1), run_idx] = 1
            run_effect = pd.DataFrame(run_effect, columns=['run-%02d'%(i+1) for i in range(n_run)])
            design_matrix = pd.concat([design_matrix.reset_index(drop=True), run_effect], 
                                    ignore_index=True, axis=1)  
            # perform GLM
            reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_sess[:, :59412])
            beta_clean = reg.coef_[:, :120].transpose(1,0).astype(np.float32)
            print('Finish performing GLM in %s %s'%(sub_name, run_name))

        elif glm_mode == 'separate':
            #####
            # GLM in a separate model(by run)
            frame_times = np.arange(n_tr) * tr 
            beta_clean = np.zeros((n_run, 120, 59412), dtype=np.float32)
            # loop in one subject
            dtseries = np.zeros((n_tr, 91282))
            onset = np.zeros((n_event))
            duration = np.repeat(0.5, n_event)
            for run_idx, file in enumerate(events_file): 
                # define run name
                run_split = file.split('_')
                run_name = '_'.join(run_split[1:3]) + '_run-' + str(int(run_split[3].split('-')[-1]))
                # fit design matrix based on trial onset time
                events_raw = pd.read_csv(pjoin(sub_func_path, file), sep='\t')
                onset = events_raw['onset'].to_numpy() + begin_dur
                label_sub = events_raw['trial_type'].to_numpy()
                trial_type = ['image%03d'%idx for idx in label_sub]
                # load time series
                dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                print(f'load {dtseries_path}')
                # prepare design matrix
                events = pd.DataFrame({'trial_type':trial_type, 'onset':onset, 'duration':duration})
                design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
                # design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
                # perform GLM
                reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries[:, :59412])
                beta_clean_run = reg.coef_[:, :120].transpose(1,0).astype(np.float32)
                print('Finish performing GLM in %s %s'%(sub_name, run_name))
                # merge beta into containers
                beta_clean[run_idx] = beta_clean_run

    # save data
    print(f'saving...{sub_name}')
    np.save(beta_clean_path, beta_clean.astype(np.float32))

# beta_sub_path = pjoin(frac_path, f'{sub_name}_ImageNet-beta_frac.npy')
# # load pre-computed beta
# beta_sub = np.load(beta_sub_path) # shape: (len(frac_params), len(sess_names), 1000, 59412)
# beta_clean = np.zeros((beta_sub.shape[1], beta_sub.shape[2], 59412))
# frac_results = sub_specific_fraction[sub_idx]
# # search the frac 
# frac_best = np.argsort(frac_results, axis=0)[0, :]
# for voxel_idx in range(beta_sub.shape[-1]):
#     beta_clean[:, :, voxel_idx] = beta_sub[frac_best[voxel_idx], :, :, voxel_idx]
# beta_clean = beta_clean.reshape((beta_sub.shape[1]*beta_sub.shape[2], 59412))
# print('Finish sub-%02d'%(sub_idx+1))
# np.save(beta_clean_path, beta_clean.astype(np.float32))