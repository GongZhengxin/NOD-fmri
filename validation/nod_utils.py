import os
import time
import numpy as np
import subprocess
from os.path import join as pjoin
import pandas as pd
import scipy.io as sio 
import nibabel as nib
from scipy.stats import zscore
import nibabel as nib
import scipy.io as sio
from nibabel import cifti2
from dnnbrain.dnn.core import Activation


def waitsecs(t=1600):
    print('sleep now')
    time.sleep(t)
    print('wake up')

def prepare_imagenet_data(dataset_root, sub_names=None, support_path='./supportfiles',clean_code='hp128_s4'):
    # define path
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    nifti_path = f'{dataset_root}'
    if not sub_names:
        sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') ] )
    n_class = 1000
    num_ses, num_run, num_trial = 4, 10, 100 
    vox_num = 59412
    # for each subject
    for sub_idx, sub_name in enumerate(sub_names):
        print(sub_name)
        label_file = pjoin(support_path, f'{sub_name}_imagenet-label.csv')
        # check whether label exists, if not then generate  
        if not os.path.exists(label_file):
            sub_events_path = pjoin(nifti_path, sub_name)
            df_img_name = []
            # find imagenet task
            imagenet_sess = [_ for _ in os.listdir(sub_events_path) if ('imagenet' in _) and ('05' not in _)]
            imagenet_sess.sort()# Remember to sort list !!!
            # loop sess and run
            for sess in imagenet_sess:
                for run in np.linspace(1,10,10, dtype=int):
                    # open ev file
                    events_file = pjoin(sub_events_path, sess, 'func',
                                        '{:s}_{:s}_task-imagenet_run-{:02d}_events.tsv'.format(sub_name, sess, run))
                    tmp_df = pd.read_csv(events_file, sep="\t")
                    df_img_name.append(tmp_df.loc[:, ['trial_type', 'stim_file']])
            df_img_name = pd.concat(df_img_name)
            df_img_name.columns = ['class_id', 'image_name']
            df_img_name.reset_index(drop=True, inplace=True)
            # add super class id
            superclass_mapping = pd.read_csv(pjoin(support_path, 'superClassMapping.csv'))
            superclass_id = superclass_mapping['superClassID'].to_numpy()
            class_id = (df_img_name.loc[:, 'class_id'].to_numpy()-1).astype(int)
            df_img_name = pd.concat([df_img_name, pd.DataFrame(superclass_id[class_id], columns=['superclass_id'])], axis=1)
            # make path
            if not os.path.exists(support_path):
                os.makedirs(support_path)
            df_img_name.to_csv(label_file, index=False)
            print(f'Finish preparing labels for {sub_name}')
        # load sub label file
        label_sub = pd.read_csv(label_file)['class_id'].to_numpy()
        label_sub = label_sub.reshape((num_ses, n_class))
        # define beta path
        beta_sub_path = pjoin(support_path, f'{sub_name}_imagenet-beta_{clean_code}_ridge.npy')
        if not os.path.exists(beta_sub_path):
            # extract from dscalar.nii
            beta_sub = np.zeros((num_ses, num_run*num_trial, vox_num))
            for i_ses in range(num_ses):
                for i_run in range(num_run):
                    run_name = f'ses-imagenet{i_ses+1:02d}_task-imagenet_run-{i_run+1}'
                    beta_data_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                    beta_sub[i_ses, i_run*num_trial : (i_run + 1)*num_trial, :] = np.asarray(nib.load(beta_data_path).get_fdata())
            # save session beta in ./supportfiles 
            np.save(beta_sub_path, beta_sub)


def prepare_coco_data(dataset_root, sub_names=None, support_path='./supportfiles',clean_code='hp128_s4'):
    # change to path of current file
    os.chdir(os.path.dirname(__file__))
    # define path
    ciftify_path = f'{dataset_root}/derivatives/ciftify'
    # Load COCO beta for 10 subjects
    if not sub_names:
        sub_names = sorted([i for i in os.listdir(ciftify_path) if i.startswith('sub') and int(i[-2:])<=9])
    num_run = 10
    n_class = 120
    clean_code = 'hp128_s4'
    for _, sub_name in enumerate(sub_names):
        sub_data_path = f'{support_path}/{sub_name}_coco-beta_{clean_code}_ridge.npy'
        if not os.path.exists(sub_data_path):
            # extract from dscalar.nii
            run_names = [ f'ses-coco_task-coco_run-{_+1}' for _ in range(num_run)]
            sub_beta = np.zeros((num_run, n_class, 59412))
            for run_idx, run_name in enumerate(run_names):
                beta_sub_path = pjoin(ciftify_path, sub_name, 'results', run_name, f'{run_name}_beta.dscalar.nii')
                sub_beta[run_idx, :, :] = np.asarray(nib.load(beta_sub_path).get_fdata())
            # save session beta in ./supportfiles 
            np.save(sub_data_path, sub_beta)

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

def runcmd(command, verbose=0):
    ret = subprocess.run(command,shell=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            encoding="utf-8",timeout=None)
    if ret.returncode == 0 and verbose:
        print("success:",ret)
    else:
        print("error:",ret)

def prepare_train_data(sub,code='hp128_s4', metric='run',runperses=10,trlperrun=100):
    data_path = './supportfiles'
    file_name = f'{sub}_imagenet-beta_{code}_ridge.npy'
    brain_resp = np.load(pjoin(data_path, file_name))
    brain_resp = brain_resp.reshape((-1,brain_resp.shape[-1]))
    brain_resp = train_data_normalization(brain_resp, metric,runperses,trlperrun)
    return brain_resp

def make_stimcsv(sub, data_path):
    act_path = f'./supportfiles'
    save_path = './supportfiles/sub_stim'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if sub == 'coco':
        if not os.path.exists(f'{save_path}/coco.stim.csv'):
            header = ['type=image\n',f'path={data_path}/stimuli/coco\n',
                    f'title=ImageNet images in {sub}\n','data=stimID\n']
            # stim files
            sub_stim = pd.read_csv(pjoin(act_path, f'coco_images.csv'), sep=',')
            stim_files = '\n'.join(sub_stim['img'].tolist())
            with open(f'{save_path}/coco.stim.csv', 'w') as f:
                f.writelines(header)
                f.writelines(stim_files)
    else:
        header = ['type=image\n',f'path={data_path}/stimuli/\n',
                    f'title=ImageNet images in {sub}\n','data=stimID\n']
        # stim files
        sub_stim = pd.read_csv(pjoin(act_path, f'{sub}_imagenet-label.csv'), sep=',')
        # replace file name
        stim_files = '\n'.join(sub_stim['image_name'].tolist())
        with open(f'{save_path}/{sub}_imagenet.stim.csv', 'w') as f:
            f.writelines(header)
            f.writelines(stim_files)

def prepare_AlexNet_feature(sub, data_path):
    act_path = f'./supportfiles/sub_stim'
    if not os.path.exists(act_path):
        os.makedirs(act_path)
    if sub == 'coco':
        stimcsv = pjoin(act_path, f'coco.stim.csv')
        if not os.path.exists(stimcsv):
            make_stimcsv(sub, data_path)
    else:
        stimcsv = pjoin(act_path, f'{sub}_imagenet.stim.csv')
        if not os.path.exists(stimcsv):
            make_stimcsv(sub, data_path)
    feature_file_path = pjoin(act_path, f'{sub}_AlexNet.act.h5')
    if not os.path.exists(feature_file_path):
        # make sure dnnbrain have been installed
        print('activation file preparing')
        cmd = f"dnn_act -net AlexNet -layer conv1 conv2 conv3 conv4 conv5 fc1 fc2 fc3 -stim ./{stimcsv} -out ./{feature_file_path}"
        runcmd(cmd, verbose=1)
    dnn_feature = Activation()
    dnn_feature.load(feature_file_path)
    return dnn_feature

def relu(M):
    relu_M = np.zeros_like(M)
    relu_M[M>0] = M[M>0]
    return relu_M

def get_voxel_roi(voxel_indice):
    roi_info = pd.read_csv('./supportfiles/roilbl_mmp.csv',sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:,0].values))
    roi_brain = sio.loadmat('./supportfiles/MMP_mpmLR32k.mat')['glasser_MMP'].reshape(-1)
    if roi_brain[voxel_indice] > 180:
        return roi_list[int(roi_brain[voxel_indice]-181)]
    else:
        return roi_list[int(roi_brain[voxel_indice]-1)]

def get_roi_data(data, roi_name, hemi=False):
    roi_info = pd.read_csv('./supportfiles/roilbl_mmp.csv',sep=',')
    roi_list = list(map(lambda x: x.split('_')[1], roi_info.iloc[:,0].values))
    roi_brain = sio.loadmat('./supportfiles/MMP_mpmLR32k.mat')['glasser_MMP'].reshape(-1)
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

def _poly_drift(order, frame_times):
    """Create a polynomial drift matrix
    """
    order = int(order)
    pol = np.zeros((np.size(frame_times), order + 1))
    tmax = float(frame_times.max())
    for k in range(order + 1):
        pol[:, k] = (frame_times / tmax) ** k
    pol = _orthogonalize(pol)
    pol = np.hstack((pol[:, 1:], pol[:, :1]))
    return pol

def _orthogonalize(X):
    """ Orthogonalize every column of design `X` w.r.t preceding columns
    """
    if X.size == X.shape[0]:
        return X

    from scipy.linalg import pinv
    for i in range(1, X.shape[1]):
        X[:, i] -= np.dot(np.dot(X[:, i], X[:, :i]), pinv(X[:, :i]))

    return X

# add drift
def add_poly_drift(design_matrix, n_tr, n_run, tr, poly_order):
    run_drift = np.zeros((n_tr*n_run, poly_order)) 
    run_frames = np.arange(n_tr) * tr
    drift_matrix = _poly_drift(order=poly_order, frame_times=run_frames) 
    for i in range(poly_order):
        run_drift[:,i] = np.tile(drift_matrix[:,i], n_run)
    run_drift = pd.DataFrame(run_drift, columns=['drift-%02d'%(i+1) for i in range(poly_order)])
    design_matrix = pd.concat([design_matrix.reset_index(drop=True), run_drift], 
                            ignore_index=True, axis=1)
# add motion 
def add_motion_var(design_matrix, sub_fpr_path, sess_name, n_tr, n_run):
    sub_cnfd_path =pjoin(sub_fpr_path, sess_name, 'func')
    cnfd_file = sorted([i for i in os.listdir(sub_cnfd_path) if ('confounds_timeseries.tsv' in i)])
    cnfd_sess = np.zeros((int(n_tr*n_run), 6))
    for run_idx, file in enumerate(cnfd_file):
        confounds = pd.read_csv(pjoin(sub_cnfd_path, file), sep='\t')
        cnfd_sess[n_tr*run_idx:n_tr*(run_idx+1),:] = confounds.loc[:,['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].values
    sess_motion = _orthogonalize(cnfd_sess)
    sess_motion = pd.DataFrame(sess_motion, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'])
    design_matrix = pd.concat([design_matrix.reset_index(drop=True), sess_motion], 
                            ignore_index=True, axis=1)

# save nifti
def save_ciftifile(data, filename, template='./supportfiles/template.dtseries.nii'):
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

def save2cifti(file_path, data, brain_models, map_names=None, volume=None, label_tables=None):
    """
    Save data as a cifti file
    If you just want to simply save pure data without extra information,
    you can just supply the first three parameters.
    NOTE!!!!!!
        The result is a Nifti2Image instead of Cifti2Image, when nibabel-2.2.1 is used.
        Nibabel-2.3.0 can support for Cifti2Image indeed.
        And the header will be regard as Nifti2Header when loading cifti file by nibabel earlier than 2.3.0.
    Parameters:
    ----------
    file_path: str
        the output filename
    data: numpy array
        An array with shape (maps, values), each row is a map.
    brain_models: sequence of Cifti2BrainModel
        Each brain model is a specification of a part of the data.
        We can always get them from another cifti file header.
    map_names: sequence of str
        The sequence's indices correspond to data's row indices and label_tables.
        And its elements are maps' names.
    volume: Cifti2Volume
        The volume contains some information about subcortical voxels,
        such as volume dimensions and transformation matrix.
        If your data doesn't contain any subcortical voxel, set the parameter as None.
    label_tables: sequence of Cifti2LableTable
        Cifti2LableTable is a mapper to map label number to Cifti2Label.
        Cifti2Lable is a specification of the label, including rgba, label name and label number.
        If your data is a label data, it would be useful.
    """
    if file_path.endswith('.dlabel.nii'):
        assert label_tables is not None
        idx_type0 = 'CIFTI_INDEX_TYPE_LABELS'
    elif file_path.endswith('.dscalar.nii'):
        idx_type0 = 'CIFTI_INDEX_TYPE_SCALARS'
    else:
        raise TypeError('Unsupported File Format')

    if map_names is None:
        map_names = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(map_names), "Map_names are mismatched with the data"

    if label_tables is None:
        label_tables = [None] * data.shape[0]
    else:
        assert data.shape[0] == len(label_tables), "Label_tables are mismatched with the data"

    # CIFTI_INDEX_TYPE_SCALARS always corresponds to Cifti2Image.header.get_index_map(0),
    # and this index_map always contains some scalar information, such as named_maps.
    # We can get label_table and map_name and metadata from named_map.
    mat_idx_map0 = cifti2.Cifti2MatrixIndicesMap([0], idx_type0)
    for mn, lbt in zip(map_names, label_tables):
        named_map = cifti2.Cifti2NamedMap(mn, label_table=lbt)
        mat_idx_map0.append(named_map)

    # CIFTI_INDEX_TYPE_BRAIN_MODELS always corresponds to Cifti2Image.header.get_index_map(1),
    # and this index_map always contains some brain_structure information, such as brain_models and volume.
    mat_idx_map1 = cifti2.Cifti2MatrixIndicesMap([1], 'CIFTI_INDEX_TYPE_BRAIN_MODELS')
    for bm in brain_models:
        mat_idx_map1.append(bm)
    if volume is not None:
        mat_idx_map1.append(volume)

    matrix = cifti2.Cifti2Matrix()
    matrix.append(mat_idx_map0)
    matrix.append(mat_idx_map1)
    header = cifti2.Cifti2Header(matrix)
    img = cifti2.Cifti2Image(data, header)
    cifti2.save(img, file_path)