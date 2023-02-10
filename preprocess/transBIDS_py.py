#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:45:09 2020

!!!!!! Please read before:
This code uses docker nipy/heudeconv to transfer *dcom/IMA* files into BIDS type. 
Frist, the machine should have aready installed docker, 
  see:http://c.biancheng.net/view/3118.html and fllowing the tutorial,
  linux os is highly recommended.
Second, BIDS infomation see: https://bids.neuroimaging.io
More tutorial infomation about using heudeconv see: https://reproducibility.stanford.edu/bids-tutorial-series-part-2a/
@author: gongzhengxin
"""
import os
from os.path import join as pjoin
import subprocess
from tqdm import tqdm
import gc
# %%
def check_path(path, verbose=0):
    if type(path) == str:
        if not os.path.exists(path):
            os.mkdir(path)
        elif verbose==1: #bool for note
            print('Already existed {}'.format(path))
    else: print('path should be str!')
# 
def runcmd(command, verbose=0):
    ret = subprocess.run(command,shell=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            encoding="utf-8",timeout=None)
    if ret.returncode == 0 and verbose:
        print("success:",ret)
    else:
        print("error:",ret)
# %%
# get subject & session information 
# check your sessinfo.csv for addapting

with open('/nfs/m1/BrainImageNet/NaturalObject/data/bold/info/processlist.tsv','r') as info:
    information = [line.split('\t') for line in info.readlines()]
header = information[0]
content = information[1:]

# rearange into a dict
# {'name': {'date': 'sessname'; 'folder': 'foldname'}}
session_dict = {}
for line in content:
    # get useful var
    name, ID, sess, status, date = line[0], line[1], line[2], line[-2], line[-1][:8]
    if not name in session_dict.keys():
        session_dict[name] = {'folder':'sub-core{:s}'.format(ID[-2:])}
    if status == 'OK':
        if 'session' in sess: 
            sessnum = eval(sess[sess.index('session')+7])
            session_dict[name][date] = 'sess-ImageNet{:02d}'.format(sessnum)
        elif 'Test' in sess:
            session_dict[name][date] = 'sess-COCO'
        elif 'LOC' in sess:
            session_dict[name][date] = 'sess-LOC'
        else:
            session_dict[name][date] = 'sess-'+sess
    else:
        session_dict[name][date] = 'sess-others'
del line, information
gc.collect()

# prepare folders, check if not create
# !!! assignments should be adpated according to current machine
os.chdir('/nfs/m1/BrainImageNet/NaturalObject/data/')
compressed_data_folder = 'bold/orig' # targzFile is where all .tar.gz data file stored
dicom_folder, nifiti_folder = 'bold/dicom', 'bold/nifti' # where store the Dcom date & Nifiti data
# check whether compressed data files in the current working path

check_path(dicom_folder, 1)
check_path(nifiti_folder, 1)
for key,value in session_dict.items():
    sub_folder = pjoin(dicom_folder, value['folder'])
    check_path(sub_folder)

# %%
# heuristic.py should be placed in the folder of nifiti_fold/code
# then excute the next section
check_path(pjoin(nifiti_folder, "code"))
if 'heuristic.py' in  os.listdir(pjoin(nifiti_folder, "code")):
    print('Yes')
else:
    raise AssertionError("Please check the heuristic.py! It should be palced in 'nifti_fold/code'!")
# %%
# get all the .tar.gz files
targzFiles = [line for line in os.listdir(compressed_data_folder) if '.tar.gz' in line] 
for file in tqdm(targzFiles[:]):
    # get the date & name information
    date, name = file.split('_')[0], file.split('_')[-1].replace('.tar.gz','') 
    # prepare all the foldnames
    try:
        target_folder = pjoin(dicom_folder,session_dict[name]['folder']) # sub folder ./sub-core0x 
        # after decompressing
        decompressed_foldname = file.replace('.tar.gz','') # where store IMA files
        session_foldername = session_dict[name][date] # standard fildname: sess-label0x
        trgname = pjoin(target_folder, session_foldername)
        final_fold = pjoin(nifiti_folder,session_dict[name]['folder'],session_foldername)
    except KeyError:
        print('%s-%s NOT IN sessinfo.tsv, pass processing' % (name, date) )
        continue
    if not os.path.exists(final_fold): # if the final fold is empty then run next
        if not os.path.exists(trgname): # if decompressed files are not existed
            # First, decompress the .tar.gz to target sub folder
            decompress_cmd = "tar -xzvf ./{:s} -C ./{:s}".format(pjoin(compressed_data_folder,file), target_folder)
            runcmd(decompress_cmd)
            
            # Second, rename the fold
            os.rename(pjoin(target_folder, decompressed_foldname), trgname)
        else: 
            print('have deteced sourcedata: {:s}'.format(trgname))
        # Third, generate .heudeconv fold, 
        # !!! this should compitable to sub folder & sess folder name
        # $$$$ docker command line
#        base_fold = "/nfs/m1/BrainImageNet/fMRIData"
#        dcom_files = "/base/"+dicom_folder+"/sub-{subject}/sess-{session}/*.IMA"
#        nifi_fold = "/base/"+ nifiti_folder
#        subID, sessID = session_dict[name]['folder'].replace("sub-", ''), session_foldername.replace("sess-", '')
#        gen_heu_cmd = "docker run --rm -i -v {:s}:/base nipy/heudiconv:latest -d {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite".format(base_fold,
#                dcom_files, nifi_fold, subID, sessID)
        
        # !!! $$$$ local command line
        base_fold = "/nfs/m1/BrainImageNet/NaturalObject/data"
        dcom_files = "$base/"+dicom_folder+"/sub-{subject}/sess-{session}/*.IMA"
        nifi_fold = "$base/"+ nifiti_folder
        subID, sessID = session_dict[name]['folder'].replace("sub-", ''), session_foldername.replace("sess-", '')
        gen_heu_cmd = "base=\"{:s}\"; heudiconv -d {:s} -o {:s} -f convertall -s {:s} -ss {:s} -c none --overwrite".format(base_fold, dcom_files, nifi_fold, subID, sessID)
        
        #runcmd(gen_heu_cmd)
        
        # Last, heuristic.py should be stored at the Nifitifolder/code
        # !!! $$$$ docker command line
        # heuristicpy = "/"+nifiti_folder+"/code/heuristic.py"
        # decom2bids_cmd1 = "docker run --rm -i -v {:s}:/base nipy/heudiconv:latest -d {:s} -o {:s}".format(base_fold, dcom_files, nifi_fold)
        # decom2bids_cmd2 = " -f /base{:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite".format(heuristicpy, subID, sessID)
        # decom2bids_cmd = decom2bids_cmd1 + decom2bids_cmd2
        
        # !!! $$$$ local command lin
        heuristicpy = "/"+nifiti_folder+"/code/heuristic.py"
        decom2bids_cmd1 = "base=\"{:s}\"; heudiconv -d {:s} -o {:s}".format(base_fold, dcom_files, nifi_fold)
        decom2bids_cmd2 = " -f $base{:s} -s {:s} -ss {:s} -c dcm2niix -b --overwrite".format(heuristicpy, subID, sessID)
        decom2bids_cmd = decom2bids_cmd1 + decom2bids_cmd2
        
#        runcmd(decom2bids_cmd)
#        
    else:
        print('{:s} already existed!'.format(trgname))

    
# %%
import os, subprocess
import time
from email.mime.text import MIMEText
from email.header import Header
from smtplib import SMTP_SSL


def send_mail(receiver, mail_title, mail_content):
    host_server = 'smtp.qq.com'
    sender_qq = '1045418215@qq.com'
    pwd = 'dflxxfxcrwkybfeg'
    # ssl
    smtp = SMTP_SSL(host_server)
    # set_debuglevel()
    smtp.set_debuglevel(1)
    smtp.ehlo(host_server)
    smtp.login(sender_qq, pwd)
    msg = MIMEText(mail_content, "plain", 'utf-8')
    msg["Subject"] = Header(mail_title, 'utf-8')
    msg["From"] = sender_qq
    msg["To"] = receiver
    smtp.sendmail(sender_qq, receiver, msg.as_string())
    smtp.quit()

def runcmd(command, verbose=0):
    ret = subprocess.run(command,shell=True,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            encoding="utf-8",timeout=None)
    if ret.returncode == 0 and verbose:
        print("success:",ret)
    else:
        print("error:",ret)

os.chdir('/nfs/m1/BrainImageNet/NaturalObject/data/bold/derivatives')
runs = [('02','LOC','retinotopy','3'),('03','COCO','fixcolor','4'),\
        ('03','LOC','category','1'), ('02','LOC','category','1'),('03','ImageNet04','object','4')]
try:
    for _ in runs:
        sub, ses, task, run = _[0], _[1], _[2], _[3]
        if not os.path.exists('melodic/sub-core{}/ses-{}'.format(sub,ses)):
            mkcmd = 'mkdir -p ./melodic/sub-core{}/ses-{}'.format(sub, ses)
            print(mkcmd)
            runcmd(mkcmd)
        a = "melodic -i ./fmriprep/sub-core{0}/ses-{1}/func/sub-core{0}_ses-{1}_task-{2}_run-{3}_space-T1w_desc-preproc_bold.nii.gz"\
            .format(sub,ses,task,run)
        b = " -o ./melodic/sub-core{0}/ses-{1}/sub-core{0}_ses-{1}_task-{2}_run-{3}.ica -v --nobet"\
            .format(sub,ses,task,run)
        c = " --bgthreshold=1 --tr=2 --mmthresh=0.5 -d 0 --report"
        cmd = a+b+c
        print("[node] {}".format(cmd))
        runcmd(cmd)
except Exception:
    
    send_mail('1045418215@qq.com','!!Error occured', 'melodic')
send_mail('1045418215@qq.com','melodic OK', 'melodic done')

# %%
import nibabel as nib 
import scipy.io as sio

label_data = nib.load('/nfs/p1/atlases/multimodal_glasser/surface/MMP_mpmLR32k.dlabel.nii').get_fdata()









