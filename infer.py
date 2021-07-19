import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tqdm
import argparse
from ymlib.debug_function import *


def parse_args():
    parser = argparse.ArgumentParser(description='inference image')
    parser.add_argument('-i', '--test-image-dir',
                        help='image test dir', required=True)
    parser.add_argument('-o', '--output-dir',
                        help='image save dir', required=True)
    parser.add_argument('--continue-test', action='store_true',
                        help='skip existing file.')
    args = parser.parse_args()
    return args


def path_decompose(path):
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    ext = os.path.splitext(path)[-1][1:]
    basename = os.path.splitext(basename)[0]
    return dirname, basename, ext


if __name__ == "__main__":
    args = parse_args()

    for filepath in tqdm.tqdm(glob.glob(os.path.join(args.test_image_dir, "*[jpg,png,jpgerr]"))):
        dirname, basename, ext = path_decompose(filepath)

