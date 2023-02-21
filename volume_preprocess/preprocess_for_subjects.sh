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
