import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import tqdm
import argparse


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


if __name__ == "__main__":
    pass

