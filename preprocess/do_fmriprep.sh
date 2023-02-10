home=/nfs/m1/BrainImageNet/fMRIData
bids_fold=$home/rawdata
out_dir=$home/derivatives_delete
work_dir=$home/workdir
license_file=/usr/local/neurosoft/freesurfer/license.txt

fmriprep-docker $bids_fold $out_dir participant \
    --participant-label core02 \
    --fs-license-file $license_file \
    --output-space anat MNI152NLin2009cAsym:res-2 fsnative fsLR \
    --use-aroma \
    --cifti-output 91k \
    -w $work_dir
fmriprep-docker $bids_fold $out_dir participant --skip-bids-validation --participant-label core02 --bids-filter-file $home/Filter.json --fs-license-file $license_file --output-spaces anat MNI152NLin6Asym:res-2 fsLR --cifti-output 91k -w $work_dir
