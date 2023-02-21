from nibabel.cifti2 import cifti2
import os
import nibabel as nib
import numpy as np
import pandas as pd
from os.path import join as pjoin

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
def save_ciftifile(data, filename, template):
    ex_cii = nib.load(template)
    if data.ndim == 1:
      data = data[None,:]
    ex_cii.header.get_index_map(0).number_of_series_points = data.shape[0]
    nib.save(nib.Cifti2Image(data,ex_cii.header), filename)

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