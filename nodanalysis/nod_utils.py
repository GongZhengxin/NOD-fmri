import os
import time
import numpy as np
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio 
import nibabel as nib
from scipy.stats import zscore
import nibabel as nib
import scipy.io as sio
from dnnbrain.dnn.core import Activation



def waitsecs(t=1600):
    print('sleep now')
    time.sleep(t)
    print('wake up')

def load_mask(sub,mask_suffix):
    mask_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/code/nodanalysis/voxel_masks'
    mask_file_path = pjoin(mask_path, f'{sub}_{mask_suffix}_fullmask.npy')
    return np.load(mask_file_path)

def train_data_normalization(data, metric='run',runperses=10,trlperrun=100):
    if data.ndim != 2:
        raise AssertionError('check data shape into (n-total-trail,n-brain-size)')
        return 0
    if metric == 'run':
        nrun = data.shape[0] / trlperrun
        for i in range(int(nrun)):
            # run normalization is to demean the run effect 
            data[i*trlperrun:(i+1)*trlperrun,:] = zscore(data[i*trlperrun:(i+1)*trlperrun,:], None)
    elif metric =='session':
        nrun = data.shape[0] / trlperrun
        nses = nrun/runperses
        for i in range(int(nses)):
            data[i*trlperrun*runperses:(i+1)*trlperrun*runperses,:] = zscore(data[i*trlperrun*runperses:(i+1)*trlperrun*runperses,:], None)
    elif metric=='trial':
        data = zscore(data, axis=1)
    return data

def prepare_train_data(sub,code='clean', metric='run',runperses=10,trlperrun=100):
    data_path = f'/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/beta/{sub}'
    file_name = f'{sub}_imagenet-beta_{code}_ridge.npy'
    brain_resp = np.load(pjoin(data_path, file_name))
    brain_resp = train_data_normalization(brain_resp, metric,runperses,trlperrun)
    return brain_resp

def prepare_AlexNet_feature(sub):
    feature_path = '/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/code/nodanalysis/sub_stim_csv'
    feature_file = f'{sub}_AlexNet.act.h5'
    dnn_feature = Activation()
    dnn_feature.load(pjoin(feature_path,feature_file))
    return dnn_feature

def relu(M):
    relu_M = np.zeros_like(M)
    relu_M[M>0] = M[M>0]
    return relu_M

def get_voxel_roi(voxel_indice):
    roi_info = pd.read_csv('/nfs/z1/zhenlab/BrainImageNet//Analysis_results/roilbl_mmp.csv',sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:,0].values))
    roi_brain = sio.loadmat('/nfs/z1/zhenlab/BrainImageNet/Analysis_results/MMP_mpmLR32k.mat')['glasser_MMP'].reshape(-1)
    if roi_brain[voxel_indice] > 180:
        return roi_list[int(roi_brain[voxel_indice]-181)]
    else:
        return roi_list[int(roi_brain[voxel_indice]-1)]

def get_roi_data(data, roi_name, hemi=False):
    roi_info = pd.read_csv('/nfs/z1/zhenlab/BrainImageNet//Analysis_results/roilbl_mmp.csv',sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:,0].values))
    roi_brain = sio.loadmat('/nfs/z1/zhenlab/BrainImageNet/Analysis_results/MMP_mpmLR32k.mat')['glasser_MMP'].reshape(-1)
    if data is not None:
      if data.shape[1] == roi_brain.size:
        if not hemi:
            return np.hstack((data[:, roi_brain==(1+roi_list.index(roi_name))], data[:, roi_brain==(181+roi_list.index(roi_name))]))
        elif hemi == 'L':
            return data[:, roi_brain==(1+roi_list.index(roi_name))]
        elif hemi == 'R':
            return data[:, roi_brain==(181+roi_list.index(roi_name))]
      else:
        roi_brain = np.pad(roi_brain, (0, data.shape[1]-roi_brain.size), 'constant')
        if not hemi:
            return np.hstack((data[:, roi_brain==(1+roi_list.index(roi_name))], data[:, roi_brain==(181+roi_list.index(roi_name))]))
        elif  hemi == 'L':
            return data[:, roi_brain==(1+roi_list.index(roi_name))]
        elif hemi == 'R':
            return data[:, roi_brain==(181+roi_list.index(roi_name))]
    else:
      roi_brain = np.pad(roi_brain, (0, 91282-roi_brain.size), 'constant')
      if type(roi_name)==list:
        return np.sum([get_roi_data(None, _,hemi) for _ in roi_name], axis=0)
      else:
        if not hemi:
            return (roi_brain==(1+roi_list.index(roi_name))) +(roi_brain==(181+roi_list.index(roi_name)))
        elif  hemi == 'L':
            return roi_brain==(1+roi_list.index(roi_name))
        elif hemi == 'R':
            return roi_brain==(181+roi_list.index(roi_name))



# save nifti
def save_ciftifile(data, filename, template='./template.dtseries.nii'):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

def solve_GMM_eq_point(m1,m2,std1,std2):
    a = np.squeeze(1/(2*std1**2) - 1/(2*std2**2))
    b = np.squeeze(m2/(std2**2) - m1/(std1**2))
    c = np.squeeze(m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1))
    return np.roots([a,b,c])

# Receptive field model
from sklearn.base import BaseEstimator, TransformerMixin
class ReceptiveFieldProcessor(BaseEstimator, TransformerMixin):
    """
    The method is to process feature map data.
    This is a transformer based on sk-learn base.

    parameter
    ---------
    center_x : float / array
    center_y : float / array
    size : float / array

    return
    ------
    feature_vector :
    """

    def __init__(self, vf_size, center_x, center_y, rf_size):
        self.vf_size = vf_size
        self.center_x = center_x
        self.center_y = center_y
        self.rf_size = rf_size

    def get_spatial_kernel(self, kernel_size):
        """
        For an image stimuli cover the visual field of **vf_size**(unit of deg.) with
        height=width=**max_resolution**, this method generate the spatial receptive
        field kernel in a gaussian manner with center at (**center_x**, **center_y**),
        and sigma **rf_size**(unit of deg.).

        parameters
        ----------
        kernel_size : int
            Usually the origin stimuli resolution.

        return
        ------
        spatial_kernel : np.ndarray

        """
        # t3 = time.time()
        # prepare parameter for np.meshgrid
        low_bound = - int(self.vf_size / 2)
        up_bound = int(self.vf_size / 2)
        # center at (0,0)
        x = np.linspace(low_bound, up_bound, kernel_size)
        y = np.linspace(low_bound, up_bound, kernel_size)
        y = -y  # adjust orientation
        # generate grid
        xx, yy = np.meshgrid(x, y)
        # prepare for spatial_kernel
        coe = 1 / (2 * np.pi * self.rf_size ** 2)  # coeficient
        ind = -((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2) / (2 * self.rf_size ** 2)  # gaussian index

        spatial_kernel = coe * np.exp(ind)  # initial spatial_kernel
        # normalize
        spatial_kernel = spatial_kernel / (np.sum(spatial_kernel)+ 1e-8)
        k = 0
        while 1:
            if len(np.unique(spatial_kernel)) == 1:
                k = k+1
                coe = 1 / (2 * np.pi * ((2**k)*self.rf_size) ** 2)
                ind = -((xx - self.center_x) ** 2 + (yy - self.center_y) ** 2) / (2 * ((2**k)*self.rf_size) ** 2)
                spatial_kernel = coe * np.exp(ind)
                spatial_kernel = spatial_kernel / (np.sum(spatial_kernel)+ 1e-8)
            else:
                break
        # t4 = time.time()
        # print('get_spatial_kernel() consumed {} min'.format((t4-t3)/60))
        return spatial_kernel

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, ):
        """

        """
        # initialize
        # t0 = time.time()
        feature_vectors = np.array([])
        if not ((type(X) == list) ^ (type(X) == np.ndarray)):
            raise AssertionError('Data type of X is not supported, '
                                 'Please check only list or numpy.ndarray')
        elif type(X) == np.ndarray:
            # input array height
            map_size = X.shape[-1]
            kernel = self.get_spatial_kernel(map_size)
            feature_vectors = np.sum(X * kernel, axis=(2, 3))
        # t1 = time.time()
        # print('transform comsumed {} min'.format((t1-t0)/60))
        return feature_vectors

class brain_data_prep():
    def __init__(self):
        ...
    def load_data(self):
        ...
    def scale_data(self):
        ...
    def mask_data(self):
        ...
    def prepare(self):
        ...

