# BIDS Data structure transformation
python data2bids.py /nfs/z1/zhenlab/BrainImageNet/NaturalObject scaninfo_second.xlsx -q ok --overwrite  

# fMRIprep preprocessing 
export prjdir=/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold
export bids_fold=$prjdir/nifti
export out_dir=$prjdir/derivatives
export work_dir=$prjdir/workdir
export license_file=/usr/local/neurosoft/freesurfer/license.txt

fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label 12 13 14 16 17 18 20 21 22 24 25 26 27 28 30 31 32 33 34 07 36 --fs-license-file $license_file --output-spaces anat fsnative -w $work_dir

# Ciftify: from the volume space to surface space
# ciftify_recon_all
# Remember to change a few codes in 'ciftify_recon_all' before running!!!!
#   Line 495: if 'v6.' in fs_version:
#   Line 594: add '-nc' between T1w_nii and freesurfer_mgz
export SUBJECTS_DIR=/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/freesurfer 
export CIFTIFY_WORKDIR=/nfs/z1/zhenlab/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify/
ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-08 sub-04
python run_cmd.py /nfs/z1/zhenlab/BrainImageNet/NaturalObject/ -c "ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-<subject>" -s 08 04 05 06 09 01

# ciftify subject fmri: process the volume data in surface space. This will make output in MNINonLinear Results folder
# if ciftify_recon_all doesn't change. This will make error output
python ciftify_subject_fmri.py
