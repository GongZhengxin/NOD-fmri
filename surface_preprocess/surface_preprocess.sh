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