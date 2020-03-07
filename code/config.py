import os, time
import sys
import numpy as np
import tensorflow as tf
from utils import *
from numpy.linalg import pinv
import glob
import shutil
import pymesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))


class Config():
    def __init__(self, args, istraining=True):

        timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
        self.resultmin = -0.95
        self.resultmax = 0.95

        self.n_epoch_Vae = int(args.epoch_VAE)
        self.n_epoch_Metric_1 = int(args.epoch_M1)
        self.n_epoch_Gan = int(args.epoch_GAN)

        self.hidden_dim = int(args.hidden_dim)

        self.featurefile = os.path.join('data', args.A + args.B + '.mat')
        self.featurefile_a = args.A + '.mat'
        self.featurefile_b = args.B + '.mat'
        self.lightfeildmat = args.A + args.B + 'lfd.mat'

        self.speed = float(args.lr)
        #print(args.useS)
        self.useS = bool(int(args.useS))
        #print(self.useS)
        self.use_tanh = bool(int(args.use_tanh))
        self.use_sigmoid = args.use_sigmoid

        self.vae_ablity = float(args.trcet)
        self.G_mapping = float(args.Gmapp)
        self.jointly = bool(int(args.joint))
        self.layer = 2
        self.sp = 'tanh'

        self.output_dir = args.output_dir
        self.control_idx = [0]

        self.meshdata_a = Datamesh(self.featurefile, self.resultmin, self.resultmax, useS=self.useS, dataname=args.A)

        self.meshdata_b = Datamesh(self.featurefile, self.resultmin, self.resultmax, useS=self.useS, dataname=args.B)

        self.lf_matrix, self.lf_matrix_min, self.lf_matrix_max, self.metric_lz_a, self.matric_lz_b = load_lfd(
            self.featurefile, 1.95, 0.05, dataname=args.A + args.B + 'lfd')

        L_a = self.meshdata_a.laplacian_matrix
        L_b = self.meshdata_b.laplacian_matrix

        for i in self.control_idx:
            temp = np.zeros((1, self.meshdata_a.pointnum1))
            temp[0, i] = 1
            L_a = np.concatenate((L_a, temp))
        self.meshdata_b.control_idx = self.control_idx
        for i in self.control_idx:
            temp = np.zeros((1, self.meshdata_b.pointnum1))
            temp[0, i] = 1
            L_b = np.concatenate((L_b, temp))
        self.meshdata_a.control_idx = self.control_idx

        self.deform_reconmatrix_a = self.meshdata_a.reconmatrix
        self.meshdata_a.deform_reconmatrix = self.deform_reconmatrix_a
        self.deform_reconmatrix_b = self.meshdata_b.reconmatrix
        self.meshdata_b.deform_reconmatrix = self.deform_reconmatrix_b


        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir + '/code' + timecurrent):
            os.makedirs(self.output_dir + '/code' + timecurrent)
        [os.system('cp ' + file + ' %s' % self.output_dir + '/code' + timecurrent + '/\n') for file in
         glob.glob(r'./code/*.py')]

        if os.path.exists(os.path.join(self.output_dir, 'log.txt')):
            self.flog = open(os.path.join(self.output_dir, 'log.txt'), 'a')
            printout(self.flog, 'add ' + timecurrent)
        else:
            self.flog = open(os.path.join(self.output_dir, 'log.txt'), 'w')

        self.iddat_name = os.path.join('data', 'id.dat')
        self.train_id_a, self.valid_id_a, self.train_id_b, self.valid_id_b = spilt_dataset(len(self.meshdata_a.feature),
                                                                                           len(self.meshdata_b.feature),
                                                                                           self.vae_ablity,
                                                                                           self.iddat_name)
        printout(self.flog, 'Train ID data A:')
        printout(self.flog, str(self.train_id_a))
        printout(self.flog, 'Valid ID data A:')
        printout(self.flog, str(self.valid_id_a))
        printout(self.flog, 'Train ID data B:')
        printout(self.flog, str(self.train_id_b))
        printout(self.flog, 'Valid ID data B:')
        printout(self.flog, str(self.valid_id_b))

        if not istraining:
            pass
        else:
            argpaser2file(args, args.output_dir + '/' + timecurrent + '.ini')

