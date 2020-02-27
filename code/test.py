import argparse
import os
import pickle
import random
import time
import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
from six.moves import xrange
import merging as modelvae
from config import *
from utils import *

timecurrent = time.strftime('%m%d%H%M', time.localtime(time.time())) + '_' + str(random.randint(1000, 9999))
test = True
parser = argparse.ArgumentParser()

parser.add_argument('--A', default='scape', type=str)
parser.add_argument('--B', default='scape', type=str)
parser.add_argument('--trcet', default=1.0, type=float)

parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--Gmapp', default=10.0, type=float)
parser.add_argument('--batch_size', default=128, type=int)

parser.add_argument('--use_sigmoid', choices=["1", "1s", "2", "2s"], default='1', type=str)
parser.add_argument('--joint', default=0, type=int)
parser.add_argument('--use_tanh', default=1, type=int)
parser.add_argument('--useS', default=0, type=int)
parser.add_argument('--epoch_VAE', default=8000, type=int)
parser.add_argument('--epoch_M1', default=10000, type=int)
parser.add_argument('--epoch_GAN', default=1000, type=int)
parser.add_argument('--output_dir', default='./result' + timecurrent, type=str)

args = parser.parse_args()

ininame = getFileName(args.output_dir, '.ini')

args = inifile2args(args, os.path.join(args.output_dir, ininame[0]))

[print('{}: {}'.format(x,k)) for x,k in vars(args).items()]

datainfo = Config(args, False)

model = modelvae.convMESH(datainfo)

with tf.device('/cpu:0'):
    # model.test_vae(datainfo)
    model.test_gan(datainfo)
