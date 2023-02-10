import nibabel as nib
import os
import subprocess
from os.path import join as pjoin
import numpy as np
import subprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'
cft_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify/'
fmriprep_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'

ses_flag  = ['Retinotopy']
file_flag = ['Atlas.dtseries']
sub_flag = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i.split('-')[-1])<=10 and int(i.split('-')[-1])!=7])
# sub_flag = ['sub-08']

Atlas_files = []
sub_dirs = [pjoin(cft_path, _, 'MNINonLinear/Results/') for _ in os.listdir(cft_path) if _ in sub_flag ]
for sub_dir in sub_dirs:
  ses_dirs = [ _ for _ in os.listdir(sub_dir) if any([__ in _ for __ in ses_flag])]
  for ses_dir in ses_dirs:
    file = [_ for _ in os.listdir(pjoin(sub_dir, ses_dir)) if any([__ in _ for __ in file_flag])]
    Atlas_files.extend([pjoin(sub_dir, ses_dir, _) for _ in file])
Atlas_files.sort()

Atlas_files = [_ for _ in Atlas_files if ('denoise' not in _) and ('discard' not in _)]

for file in tqdm(Atlas_files):
  print(file)
  sub_dir = file[file.find('sub'):(file.find('sub')+6)]
  L_hemi = pjoin(cft_path, sub_dir, 'MNINonLinear/fsaverage_LR32k', f'{sub_dir}.L.midthickness.32k_fs_LR.surf.gii')
  R_hemi = pjoin(cft_path, sub_dir, 'MNINonLinear/fsaverage_LR32k', f'{sub_dir}.R.midthickness.32k_fs_LR.surf.gii')
  # input_file = file.replace('Atlas_s0', 'Atlas_clean')
  input_file = file
  fwhm = 4
  output_file = file.replace('Atlas', f'Atlas_s{fwhm}')
  if os.path.exists(output_file):
    continue
  else:
    sigma = fwhm/np.sqrt(8*np.log(2))
    direction = 'COLUMN'
    cmd_head = 'wb_command -cifti-smoothing '
    L_cmd = f'-left-surface {L_hemi} '
    R_cmd = f'-right-surface {R_hemi} '
    smooth_cmd = cmd_head +f'{input_file} {sigma} {sigma} {direction} {output_file} ' + L_cmd + R_cmd
    print(smooth_cmd)
    subprocess.check_call(smooth_cmd, shell=True)