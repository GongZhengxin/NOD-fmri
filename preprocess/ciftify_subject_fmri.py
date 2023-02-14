import os
import subprocess
from os.path import join as pjoin
from tqdm import tqdm

#%%
ncpu = input('Cpu num:')
#start = input('Start index:')
#end = input('End index:')

# %%

# input path includes all subject folds
input_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'

# flags for selection
sub_flag = ['sub-01']#, 'sub-02','sub-04','sub-05','sub-06'
ses_flag = ['ses-ImageNet01']
mod_flag = ['func']
file_flag = ['space-T1w_desc-preproc_bold_hp128.nii']

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

#retino_func_files = [_ for _ in func_files if 'retinotopy' in _]
#other_func_files = list(set(func_files) - set(retino_func_files))

#%% run cml
#ciftify_work_dir = '/nfs/m1/BrainImageNet/NaturalObject/data/bold/Analysis_derivatives/ciftify'
cmd_export = 'export CIFTIFY_WORKDIR=/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify;'
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














