#!/bin/bash
export SUBJECTS_DIR=/nfs/m1/BrainImageNet/NaturalObject/data/bold/derivatives/freesurfer 
export CIFTIFY_WORKDIR=/nfs/m1/BrainImageNet/NaturalObject/data/bold/derivatives/ciftify

ciftify_recon_all --surf-reg MSMSulc --resample-to-T1w32k sub-04
		
		 

