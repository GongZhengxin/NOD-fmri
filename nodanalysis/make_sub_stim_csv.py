# template
import os
import numpy as np
from os.path import join as pjoin
import pandas as pd

for i, sub in enumerate(['sub-{:02d}'.format(_) for _ in [1,2,3,5,6,9,10]]):
    act_path = f'./supportfiles'
    header = ['type=image\n','path=/nfs/z1/zhenlab/DNN/ImgDatabase/ImageNet_2012/ILSVRC2012_img_train\n',
                f'title=ImageNet images in {sub}\n','data=stimID\n']
    # stim files
    sub_stim = pd.read_csv(pjoin(act_path, f'{sub}_imagenet-label.csv'), sep=',')
    # replace file name
    stim_files = []
    for stim in sub_stim['image_name']:
        folder = stim.split('_')[0]
        stim_files.append(f'{folder}/{stim}\n') 
    save_path = './supportfiles/sub_stim'
    with open(f'{save_path}/{sub}_imagenet.stim.csv', 'w') as f:
        f.writelines(header)
        f.writelines(stim_files)