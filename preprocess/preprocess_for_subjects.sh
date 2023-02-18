# >>fMRIprep preprocessing 
# define your project path to **prjdir**
export prjdir='PATHtoDataset'
export bids_fold=$prjdir/
export out_dir=$prjdir/derivatives
export work_dir= 'PATHtoWorkdir' # note that this dir cannot be set under the bids folder
# set your freesurfer license path, if file not exists
#  please register in https://surfer.nmr.mgh.harvard.edu/registration.html
# after recieving the file, set the path
export license_file=/usr/local/neurosoft/freesurfer/license.txt

# fmriprep in docker
# add/delete participants following --participant-label
# for more usages to view https://fmriprep.org/en/stable/
fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label 01 02 03 --fs-license-file $license_file --output-spaces anat fsnative -w $work_dir

# if anat derivatives exists, to save time, anatomical pipelines can be skipped, 
# it seems that only single participant can be processed in this way:
fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label 01 --fs-license-file $license_file --output-spaces anat -w $work_dir --anat-derivatives $prjdir/derivatives/fmriprep/sub-09/ses-anat/anat


# >>Ciftify: from the volume space to surface space
# ciftify_recon_all
# Remember to change a few codes in 'ciftify_recon_all' before excuting!!!!
#   Line 495: if 'v6.' in fs_version:
#   Line 594: add '-nc' between T1w_nii and freesurfer_mgz
export prjdir=/nfs/z1/userhome/GongZhengXin/NVP/forfmriprep
export SUBJECTS_DIR=$prjdir/derivatives/freesurfer 
export CIFTIFY_WORKDIR=$prjdir/derivatives/ciftify/

# process specific subject
ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-09

# batch processing
python run_cmd.py $prjdir -c "ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-<subject>" -s 01 02 03 04

# ciftify subject fmri: process the volume data in surface space. This will make output in MNINonLinear Results folder
# if ciftify_recon_all doesn't change. This will make error output
# !! variables in ciftify_subject_fmri need to be changed before excute
python ciftify_subject_fmri.py
