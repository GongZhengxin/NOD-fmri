import nibabel as nib
import os
from os.path import join as pjoin
from nibabel import Nifti1Header
from nibabel import Nifti1Image
import numpy as np
import subprocess
import pandas as pd
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

# save nifti
def save_ciftifile(data, filename, template):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

beta_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta'
cft_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify/'
fmriprep_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/fmriprep'

ses_flag  = ['COCO']
file_flag = ['Atlas_s0.dtseries']
sub_flag = sorted([i for i in os.listdir(beta_path) if i.startswith('sub') and int(i.split('-')[-1])<=10])
# sub_flag = ['sub-10']

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
  if not os.path.exists(file.replace('Atlas_s0', 'Atlas_clean')):
    img = nib.load(file)
    data = img.get_fdata()[:]
    dataTdim, dataXdim = data.shape[0], data.shape[1]
    datanewZdim = np.ceil(dataXdim/100)
    # transpose and demean
    mean_data = data.mean(axis=0)
    data = np.r_[(data - mean_data).transpose(), np.zeros((int(100*datanewZdim - dataXdim), dataTdim))]
    data = data.reshape((10, 10, int(datanewZdim), dataTdim))

    # write a header
    header = Nifti1Header()
    header.set_data_shape((10, 10, int(datanewZdim), dataTdim))

    # save the data
    file_name = file.split('/')[-1]
    nii_file = file.replace(f'{file_name}', 'Atlas.nii.gz')
    nib.save(Nifti1Image(data, np.diag([1,1,1,1]), header),  nii_file)
    highpass = 128
    nii_file_path = nii_file.replace('.nii.gz', '')
    fsl_cmd = f'fslmaths {nii_file_path} -bptf {highpass/4} -1 {nii_file_path}'
    subprocess.check_call(fsl_cmd, shell=True)
    data = nib.load(pjoin(cft_path, sub_dir, nii_file)).get_fdata()[:]
    data = data.reshape((int(100*datanewZdim), dataTdim))[0:dataXdim,:].transpose() + mean_data
    # # load motion info
    # # replace old task name
    # if 'object' in file:
    #   file_tmp = file.replace('object', 'naturalvision')
    # else:
    #   file_tmp = file
    # sub_name = file_tmp.split('/')[-5]
    # run_name = file_tmp.split('/')[-2]
    # sess_name = run_name.split('_')[0]
    # confoundcsv = pjoin(fmriprep_path, sub_name, sess_name, 'func', 
    #                     '{}_{}_desc-confounds_timeseries.tsv'.format(sub_name, run_name))
    # df_conf = pd.read_csv(confoundcsv, sep='\t')
    # motion = np.vstack(tuple([np.array(df_conf[_]).astype(np.float64) for _ in \
    #                   ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']])).transpose(1, 0)
    # # regress motion here
    # reg = LinearRegression().fit(motion, data)
    # data = data - np.dot(motion, reg.coef_.transpose(1, 0))
    # data = data.astype(np.float32)
    # save cifti
    save_ciftifile(data.astype(np.float32), file.replace('Atlas_s0', 'Atlas_clean'), file)
  # else:
  file_name = file.split('/')[-1]
  nii_file = file.replace(f'{file_name}', 'Atlas.nii.gz')
  if os.path.exists(nii_file): 
    rm_cmd = f'rm -f {nii_file}'
    subprocess.check_call(rm_cmd, shell=True)
    print(rm_cmd)

