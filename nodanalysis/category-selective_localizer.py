import os
import numpy as np
import pandas as pd
import nibabel as nib
from os.path import join as pjoin
from nod_utils import save2cifti, _orthogonalize
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast

# change to path of current file
os.chdir(os.path.dirname(__file__))
# define path
dataset_root = '/nfs/z1/userhome/GongZhengXin/NVP/data_upload/NOD'#'PATHtoDataset'
fmriprep_path = f'{dataset_root}/derivatives/fmriprep'
ciftify_path = f'{dataset_root}/derivatives/ciftify'
nifti_path = f'{dataset_root}'
result_path = './floc_results'

# load design and perform GLM
# load nifti events and define the design matrix
# prepare params
n_event = 10
n_tr = 150
tr = 2
frame_times = np.arange(n_tr) * tr
sub_names = sorted([_ for _ in os.listdir(nifti_path) if 'sub' in _ and 'ses-floc' in os.listdir(pjoin(nifti_path, _))])

# one subject
sub_names = [_ for _ in sub_names ]

z_score_sum = np.zeros((len(sub_names), 5, 91282))
clean_code = 'hp128_s4'

for sub_id, sub_name in enumerate(sub_names):
    z_score_sub = np.zeros((1, 5, 91282))
    sess_names = sorted([i for i in os.listdir(pjoin(nifti_path, sub_name)) if 'ses-floc' in i])
    # start runnning
    for sess_id, sess_name in enumerate(sess_names):
        sess_path = pjoin(nifti_path, sub_name, sess_name, 'func')
        run_names = sorted([i for i in os.listdir(sess_path) if i.endswith('.tsv')])
        sub_cnfd_path =pjoin(fmriprep_path, sub_name, sess_name, 'func')
        cnfd_files = sorted([i for i in os.listdir(sub_cnfd_path) if ('confounds_timeseries.tsv' in i)])
        for run_id, run_name in enumerate(run_names):
            run_path = pjoin(sess_path, run_name)
            run_par = pd.read_csv(run_path, sep='\t')
            # fit design matrix based on trial onset time
            onset = run_par['onset'].to_numpy()
            trial_type = run_par['trial_type']
            duration = np.repeat(4, trial_type.shape[0])
            # prepare events
            events_used = pd.DataFrame({'trial_type':trial_type, 'onset':onset, 'duration':duration})
            # prepare motion regressors
            cnfd_file = cnfd_files[run_id]
            confounds = pd.read_csv(pjoin(sub_cnfd_path, cnfd_file), sep='\t')
            cnfd_run = confounds.loc[:,['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].values
            run_motion = _orthogonalize(cnfd_run)
            run_motion = pd.DataFrame(run_motion, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
            # prepare design matrix
            design_matrix = make_first_level_design_matrix(frame_times, events_used, hrf_model='spm',add_regs=run_motion)
            # load dtseries
            run_name = 'ses-floc_task-floc_run-%d'%(run_id+1)
            dtseries_path = pjoin(ciftify_path, sub_name, 'results/', run_name, 
                                f'{run_name}_Atlas_{clean_code}.dtseries.nii')
            if not os.path.exists(dtseries_path):
                print(f'WARNING:{run_name} dtseries file not found!')
                continue
            dtseries = nib.load(dtseries_path).get_fdata()
            # perform GLM
            labels, estimates = run_glm(dtseries, design_matrix.values)

            # generate specified localizer: from http://vpnl.stanford.edu/fLoc/
            # order: 0: adult; 1: body; 2: car; 3: child; 4: corridor; 5: house; 6: instrument; 7: limb; 8: number; 9: word
            # Character-selective regions: [word number] > [body limb child adult corridor house car instrument]
            # Body-selective regions: [body limb] > [word number child adult corridor house car instrument]
            # Face-selective regions: [child adult] > [word number body limb corridor house car instrument]
            # Place-selective regions: [corridor house] > [word number body limb child adult car instrument]
            # Object-selective regions: [car instrument] > [word number body limb child adult corridor house]
            
            contrast_matrix = np.eye(design_matrix.shape[1])
            basic_contrasts = dict([(column, contrast_matrix[i]) for i, column in enumerate([i for i in design_matrix.columns])])
            contrasts = {
                'character': (
                    (basic_contrasts['word'] + basic_contrasts['number'])/2 - (basic_contrasts['body'] + basic_contrasts['limb'] + basic_contrasts['child']
                    + basic_contrasts['adult']+ basic_contrasts['corridor']+ basic_contrasts['house']+ basic_contrasts['car']+ basic_contrasts['instrument'])/8
                ), 
                'body': (
                    (basic_contrasts['body'] + basic_contrasts['limb'])/2 - (basic_contrasts['word'] + basic_contrasts['number'] + basic_contrasts['child']
                    + basic_contrasts['adult']+ basic_contrasts['corridor']+ basic_contrasts['house']+ basic_contrasts['car']+ basic_contrasts['instrument'])/8
                ), 
                'face': (
                    (basic_contrasts['child'] + basic_contrasts['adult'])/2 - (basic_contrasts['body'] + basic_contrasts['limb'] + basic_contrasts['word']
                    + basic_contrasts['number']+ basic_contrasts['corridor']+ basic_contrasts['house']+ basic_contrasts['car']+ basic_contrasts['instrument'])/8
                ), 
                'place': (
                    (basic_contrasts['corridor'] + basic_contrasts['house'])/2 - (basic_contrasts['body'] + basic_contrasts['limb'] + basic_contrasts['child']
                    + basic_contrasts['adult']+ basic_contrasts['word']+ basic_contrasts['number']+ basic_contrasts['car']+ basic_contrasts['instrument'])/8
                ), 
                'object': (
                    (basic_contrasts['car'] + basic_contrasts['instrument'])/2 - (basic_contrasts['body'] + basic_contrasts['limb'] + basic_contrasts['child']
                    + basic_contrasts['adult']+ basic_contrasts['corridor']+ basic_contrasts['house']+ basic_contrasts['word']+ basic_contrasts['number'])/8
                ),     
            }
            z_score_run = np.zeros((5, 91282))
            for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
                # compute contrast-related statistics
                contrast = compute_contrast(labels, estimates, contrast_val, contrast_type='t')
                # we present the Z-transform of the t map
                z_score_run[index] = contrast.z_score()
            # save beta
            z_score_sub = np.concatenate((z_score_sub, z_score_run[np.newaxis, :, :]), axis=0)
            print('Finish estimating in %s %s'%(sub_name, run_name))
    z_score_sub = np.delete(z_score_sub, 0, axis=0)
    z_score_sub = np.mean(z_score_sub, axis=0)
    # get brainmodel
    temp = nib.load('./supportfiles/template.dtseries.nii')
    bm = list(temp.header.get_index_map(1).brain_models)[0:2]
    # file writing
    dscfile_path = pjoin(result_path, f'{sub_name}-{clean_code}_beta.dscalar.nii')
    stim_names = ['Character - others', 'Body - others', 'Face - others', 'Place - others', 'Object - others']
    save2cifti(file_path=dscfile_path, data=z_score_sub.astype(np.float32), brain_models=bm, map_names=stim_names)
    txt_file = pjoin(result_path, f'{sub_name}-{clean_code}_label.txt')
    stim_names = '\n'.join(stim_names)
    with open(txt_file, 'w') as f:
        f.write(stim_names)
    print(f'Finish loading {sub_name}')
