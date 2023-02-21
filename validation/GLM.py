import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.linear_model import Ridge
from os.path import join as pjoin
from nilearn.glm.first_level import make_first_level_design_matrix
from nod_utils import save2cifti, add_poly_drift, add_motion_var

# define path
dataset_path = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD'
ciftify_path = f'{dataset_path}/derivatives/ciftify'
nifti_path = f'{dataset_path}'
fprep_path = f'{dataset_path}/derivatives/fmriprep'

# prepare params
sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub')])
target = 'imagenet' # imagenet or coco
clean_code = 'hp128_s4' # version of surface data, if None the raw surface data will be used
glm_mode = 'separate' # seperate or full GLM model in coco beta estimate
motion_on = True # whether to include motion regressors
poly_order = None # whetehr to include drift regressors
alpha = 0.01

for sub_idx, sub_name in enumerate(sub_names):
    # prepare basic path
    sub_nft_path = pjoin(nifti_path, sub_name)
    sub_fpr_path = pjoin(fprep_path, sub_name)
    _result_path = pjoin(ciftify_path, sub_name, 'results/')
    # if not os.path.exists(beta_clean_path):
    if target == 'imagenet':
        # prepare params
        tr, begin_dur, n_tr, n_event = 2, 16, 256, 100
        # define beta path
        sess_names = sorted([i for i in os.listdir(sub_nft_path) if ('imagenet05' in i)]) 
        run_names = sorted([i for i in os.listdir(_result_path) if ('imagenet05' in i)])
        beta_clean = np.zeros((len(run_names)*n_event, 59412), dtype=np.float32)

        # loop in one subject
        for sess_idx, sess_name in enumerate(sess_names):
            sub_func_path = pjoin(sub_nft_path, sess_name, 'func')
            events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i ])
            n_run = len(events_file)
            # initialize
            dtseries_sess = np.zeros((int(n_tr * n_run), 91282))
            onset_sess = np.zeros((int(n_run*n_event)))
            duration_sess = np.zeros((int(n_run*n_event)))
            
            for run_idx, file in enumerate(events_file): 
                # define run name
                run_split = file.split('_')
                run_name = '_'.join(run_split[1:3]) + '_run-' + str(int(run_split[3].split('-')[-1]))
                # fit design matrix based on trial onset time
                events = pd.read_csv(pjoin(sub_func_path, file), sep='\t')
                duration = events['duration']
                onset = events['onset'].to_numpy() + begin_dur
                # load time series
                if clean_code:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                else:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                print(f'load {dtseries_path}')
                # concantenate all info into sess params
                dtseries_sess[n_tr*run_idx:n_tr*(run_idx+1)] = dtseries
                onset_sess[n_event*run_idx:n_event*(run_idx+1)] = onset + run_idx * n_tr * tr
                duration_sess[n_event*run_idx:n_event*(run_idx+1)] = duration
            # prepare design matrix
            frame_times = np.arange(n_tr*n_run) * tr 
            trial_type = ['image%04d'%(idx+1) for idx in range(n_event * n_run)]
            events = pd.DataFrame({'trial_type':trial_type, 'onset':onset_sess, 'duration':duration_sess})
            design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
            design_matrix.drop(design_matrix.columns[-1], axis=1, inplace=True)
            # add column for run effect
            run_effect = np.zeros((n_tr * n_run, n_run))
            for run_idx in range(n_run):
                run_effect[n_tr*run_idx:n_tr*(run_idx+1), run_idx] = 1
            run_effect = pd.DataFrame(run_effect, columns=['run-%02d'%(i+1) for i in range(n_run)])
            design_matrix = pd.concat([design_matrix.reset_index(drop=True), run_effect], 
                                    ignore_index=True, axis=1)
            # add poly drift
            if poly_order:
                add_poly_drift(design_matrix, n_tr, n_run, tr, poly_order)
            # add motion 
            if motion_on:
                add_motion_var(design_matrix, sub_fpr_path, sess_name, n_tr, n_run)
            # perform GLM
            reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_sess[:, :59412])
            beta = reg.coef_[:, :int(n_event*n_run)].transpose(1,0).astype(np.float32)
            if n_run==10:
                beta_clean[1000*sess_idx:1000*(sess_idx+1)] = beta
            else:
                if sess_idx > 0:
                    beta_clean[1000*sess_idx::] = beta
                else:
                    beta_clean = beta
            for run_idx, file in enumerate(events_file):
                # define run name
                run_split = file.split('_')
                run_name = '_'.join(run_split[1:3]) + '_run-' + str(int(run_split[3].split('-')[-1]))
                # fit design matrix based on trial onset time
                events = pd.read_csv(pjoin(sub_func_path, file), sep='\t')
                stim_names = [_.split('/')[-1] for _ in events['stim_file']]
                # load header - brain model
                temp = nib.load('./supportfiles/template.dtseries.nii')
                bm = list(temp.header.get_index_map(1).brain_models)[0:2]
                # data load
                run_data = beta[n_event * (run_idx) : n_event * (run_idx + 1), :]
                dsc_file = pjoin(_result_path, run_name, f'{run_name}_beta.dscalar.nii')
                save2cifti(file_path=dsc_file, data=run_data, brain_models=bm, map_names=stim_names)
                txt_file = pjoin(_result_path, run_name, f'{run_name}_label.txt')
                with open(txt_file, 'w') as f:
                    f.writelines(stim_names)
            print('Finish performing GLM in %s %s'%(sub_name, sess_name))
    elif target == 'coco':
        # define params:
        tr, begin_dur, n_tr, n_event = 2, 16, 241, 120
        sess_name = 'ses-coco'
        sub_func_path = pjoin(sub_nft_path, sess_name, 'func')
        events_file = sorted([i for i in os.listdir(sub_func_path) if 'events' in i ])
        n_run = len(events_file)

        if glm_mode == 'full':
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
                if clean_code:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                else:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas.dtseries.nii')
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
            # add poly drift
            if poly_order:
                add_poly_drift(design_matrix, n_tr, n_run, tr, poly_order)
            # add motion 
            if motion_on:
                add_motion_var(design_matrix, sub_fpr_path, sess_name, n_tr, n_run)
            # perform GLM
            reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries_sess[:, :59412])
            beta_clean = reg.coef_[:, :120].transpose(1,0).astype(np.float32)
            # define run name
            run_split = events_file[0].split('_')
            ses_name = '_'.join(run_split[1:3])
            # ceate ses folder
            if os.path.exists(pjoin(_result_path, ses_name)):
                os.makedirs(pjoin(_result_path, ses_name))
            # stimulus names
            stim_names = sorted(label_sub)
            # load header - brain model
            temp = nib.load('./supportfiles/template.dtseries.nii')
            bm = list(temp.header.get_index_map(1).brain_models)[0:2]
            # data load
            dsc_file = pjoin(_result_path, ses_name, f'{ses_name}_beta.dscalar.nii')
            save2cifti(file_path=dsc_file, data=run_data, brain_models=bm, map_names=stim_names)
            txt_file = pjoin(_result_path, run_name, f'{run_name}_label.txt')
            with open(txt_file, 'w') as f:
                f.writelines(stim_names)
            print('Finish performing GLM in %s %s'%(sub_name, ses_name))

        elif glm_mode == 'separate':
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
                if clean_code:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas_{clean_code}.dtseries.nii')
                else:
                    dtseries_path = pjoin(_result_path, run_name, f'{run_name}_Atlas.dtseries.nii')
                dtseries = nib.load(dtseries_path).get_fdata()
                print(f'load {dtseries_path}')
                # prepare design matrix
                events = pd.DataFrame({'trial_type':trial_type, 'onset':onset, 'duration':duration})
                design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, hrf_model='spm')
                # add poly drift
                if poly_order:
                    add_poly_drift(design_matrix, n_tr, n_run, tr, poly_order)
                # add motion 
                if motion_on:
                    add_motion_var(design_matrix, sub_fpr_path, sess_name, n_tr, n_run)
                # perform GLM
                reg = Ridge(alpha=alpha, fit_intercept=False).fit(design_matrix.values, dtseries[:, :59412])
                beta_clean_run = reg.coef_[:, :120].transpose(1,0).astype(np.float32)
                print('Finish performing GLM in %s %s'%(sub_name, run_name))
                # merge beta into containers
                # stimulus names
                stim_names = sorted(label_sub)
                # load header - brain model
                temp = nib.load('./supportfiles/template.dtseries.nii')
                bm = list(temp.header.get_index_map(1).brain_models)[0:2]
                # data load
                dsc_file = pjoin(_result_path, ses_name, f'{ses_name}_beta.dscalar.nii')
                save2cifti(file_path=dsc_file, data=run_data, brain_models=bm, map_names=stim_names)
                txt_file = pjoin(_result_path, run_name, f'{run_name}_label.txt')
                with open(txt_file, 'w') as f:
                    f.writelines(stim_names)
    # report
    print(f'saved...{sub_name}-{target}')
