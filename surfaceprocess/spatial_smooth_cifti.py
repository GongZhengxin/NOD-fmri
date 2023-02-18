import os
import subprocess
from os.path import join as pjoin
import numpy as np
import subprocess
from tqdm import tqdm

# define paths
dataset_path = 'PATHtoDataset'
cft_path = f'{dataset_path}/derivatives/ciftify/'
fmriprep_path = f'{dataset_path}/derivatives/fmriprep'

# flag labels
ses_flag  = ['imagenet']
file_flag = ['Atlas_hp128.dtseries']
sub_flag = sorted([i for i in os.listdir(cft_path) if i.startswith('sub')])
# # if only specific subjects are wanted:
# sub_flag = ['sub-02', 'sub-03']
# sub_flag = sorted([i for i in os.listdir(cft_path) if i.startswith('sub') and int(i.split('-')[-1])<=9])

# collect Atlas files
Atlas_files = []
sub_dirs = [pjoin(cft_path, _, 'results/') for _ in os.listdir(cft_path) if _ in sub_flag ]
for sub_dir in sub_dirs:
  ses_dirs = [ _ for _ in os.listdir(sub_dir) if any([__ in _ for __ in ses_flag])]
  for ses_dir in ses_dirs:
    file = [_ for _ in os.listdir(pjoin(sub_dir, ses_dir)) if any([__ in _ for __ in file_flag])]
    Atlas_files.extend([pjoin(sub_dir, ses_dir, _) for _ in file])
Atlas_files.sort()

Atlas_files = [_ for _ in Atlas_files]

# whether delete the input file
delete_hp = True
# spatial smooth
for file in tqdm(Atlas_files):
  print(file)
  # get the sub name
  sub_dir = file[file.find('sub'):(file.find('sub')+6)]
  # prepare surf.gii files for smooth
  L_hemi = pjoin(cft_path, sub_dir, 'standard_fsLR_surface', f'{sub_dir}.L.midthickness.32k_fs_LR.surf.gii')
  R_hemi = pjoin(cft_path, sub_dir, 'standard_fsLR_surface', f'{sub_dir}.R.midthickness.32k_fs_LR.surf.gii')

  # check the output file
  output_file = file.replace('Atlas_hp128', f'Atlas_hp128_s{fwhm}')
  #  automatical overwriting is not allowed, please manually delete existing files
  if os.path.exists(output_file):
    continue
  else:
    # prepare command inputs
    input_file = file
    fwhm = 4 
    sigma = fwhm/np.sqrt(8*np.log(2))
    direction = 'COLUMN'
    cmd_head = 'wb_command -cifti-smoothing '
    L_cmd = f'-left-surface {L_hemi} '
    R_cmd = f'-right-surface {R_hemi} '
    # full cmd
    smooth_cmd = cmd_head +f'{input_file} {sigma} {sigma} {direction} {output_file} ' + L_cmd + R_cmd
    print(smooth_cmd)
    # run cmd
    subprocess.check_call(smooth_cmd, shell=True)
    # if input is a temporary file delete it
    if delete_hp :
      rm_cmd = f'rm -f {input_file}'
      subprocess.check_call(rm_cmd, shell=True)
