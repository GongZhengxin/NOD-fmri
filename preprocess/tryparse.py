import os
import argparse
import subprocess as  sp
import pandas as pd
import numpy as np

from os.path import join as pjoin
from tqdm import tqdm

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="path fetch to scaninfo.xlsx")
parser.add_argument("-s", "--subject", type=str, nargs="+", help="subjects")
parser.add_argument("-ss", "--session", type=str, nargs="+", help="sessions")
parser.add_argument("-q", "--quality-filter", type=str, help="", choices=['ok', 'all','discard'], default='ok')
args = parser.parse_args()

print(args.file, args.subject, args.session, args.quality_filter)
if args.subject and args.session:
    print(args.sebject)
    print(args.session)
