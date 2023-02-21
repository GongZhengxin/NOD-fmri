import os
import subprocess
from os.path import join as pjoin
from tqdm import tqdm

#%%
ncpu = input('Cpu num:')

# input path includes all subject folds
prj_dir = 'PATHtoDataset'
input_path = pjoin(prj_dir, 'derivatives/fmriprep')

# flags for selection
# set subjects, sessions, files that waiting for ciftify transformation
sub_flag = ['sub-01']
ses_flag = ['ses-imagenet03','imagenet04'] # ses-prf | ses-floc | ses-coco
mod_flag = ['func']
file_flag = ['space-T1w_desc-preproc_bold.nii']

# initialize func_files
func_files = []

# generate all run files
subject_folds = [ _ for _ in os.listdir(input_path) if any([ __ in _ for __ in \
                 sub_flag]) and ('.' not in _)]
for subject in subject_folds:
    session_folds = [ _ for _ in os.listdir(pjoin(input_path, subject)) \
                     if any([ __ in _ for __ in ses_flag]) and ('.' not in _)]
    for session in session_folds:
        modality_folds = [ _ for _ in os.listdir(pjoin(input_path, subject, session)) \
                     if any([ __ in _ for __ in mod_flag]) and ('.' not in _)]
        for modality in modality_folds:
            current_files = [ _ for _ in os.listdir(pjoin(input_path, subject, session, modality)) \
                     if all([ __ in _ for __ in file_flag])]
            func_files.extend([pjoin(input_path,subject,session,modality,file) for file in \
                  current_files])

#%% run cml
cmd_export = f'export CIFTIFY_WORKDIR={prj_dir}/derivatives/ciftify;'
cmd_pre = f'ciftify_subject_fmri --surf-reg MSMSulc --n_cpus {ncpu}'
for func_file in tqdm(func_files):
    out_name = os.path.basename(func_file).split('.')[0]
    out_name = out_name.split('_') 
    subject = [i for i in out_name if i.startswith('sub-')][0]
    out_name = '_'.join([i for i in out_name
                         if i.startswith('ses-') or i.startswith('task-') or i.startswith('run-') or i.startswith('denoise-') or i.startswith('hp')])
    cmd = cmd_export + cmd_pre + ' ' + ' '.join([func_file, subject, out_name])
    print(cmd)
    subprocess.call(cmd, shell=True)














