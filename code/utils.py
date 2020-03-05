# coding: utf-8
from math import sin, cos, sqrt
from six.moves import xrange

import h5py
import numpy as np
import scipy.io as sio
import tensorflow as tf
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial.distance
import scipy.interpolate as interpolate
import pymesh, os, random, pickle, configparser, argparse


class Id:
    def __init__(self, Ia, Ib):
        self.Ia=Ia
        self.Ib=Ib
    def show(self):
        print('A: \n%s\nB: \n%s'%(self.Ia, self.Ib))
    def tofile(self, filename):
        sio.savemat(filename,{'IA':self.Ia,'IB':self.Ib})
        print('A: %s\nB: %s'%(self.Ia, self.Ib))
    def loadfile(self, filename):
        data = sio.loadmat(filename)
        self.Ia = data['Ia']
        self.Ib = data['Ib']
        print('A: %s\nB: %s'%(self.Ia, self.Ib))

class Datamesh:
    def __init__(self, path, resultmin, resultmax, useS=True, dataname = '', graphconv=False):
        data = h5py.File(path, mode = 'r')
        data = data[dataname]
        datalist = data.keys()
        self.resultmin = resultmin
        self.resultmax = resultmax

        logr = np.transpose(data['FLOGRNEW'], (2, 1, 0)).astype('float32')
        s = np.transpose(data['FS'], (2, 1, 0)).astype('float32')
        neighbour1 = np.transpose(data['neighbour1']).astype('int32')
        cotweight1 = np.transpose(data['cotweight1']).astype('float32')

		
        self.pointnum1 = neighbour1.shape[0]
        self.maxdegree1 = neighbour1.shape[1]
        self.modelnum = len(logr)

        self.logr_avg = logr.mean(axis = 0)
        self.s_avg = s.mean(axis = 0)
        logr = logr - self.logr_avg
        s = s - self.s_avg
        logrmin = logr.min()
        self.logrmin = logrmin - 1e-6
        logrmax = logr.max()
        self.logrmax = logrmax + 1e-6
        smin = s.min()
        self.smin = smin - 1e-6
        smax = s.max()
        self.smax = smax + 1e-6

        rnew = (resultmax - resultmin) * (logr - self.logrmin) / (self.logrmax - self.logrmin) + resultmin
        snew = (resultmax - resultmin) * (s - self.smin) / (self.smax - self.smin) + resultmin

        if useS:
            self.feature = np.concatenate((rnew, snew), axis=2)
        else:
            self.feature = rnew

        self.nb1 = neighbour1
        self.cotw1 = np.zeros((cotweight1.shape[0], cotweight1.shape[1], 1)).astype('float32')
        for i in range(1):
            self.cotw1[:, :, i] = cotweight1

        self.degree1 = np.zeros((neighbour1.shape[0], 1)).astype('float32')
        for i in range(neighbour1.shape[0]):
            self.degree1[i] = np.count_nonzero(self.nb1[i])

        self.laplacian_matrix = np.transpose(data['L']).astype('float32')
        self.reconmatrix = np.transpose(data['recon']).astype('float32')
        self.vdiff = np.transpose(data['vdiff'], (2, 1, 0)).astype('float32')
        self.all_vertex = np.transpose(data['vertex'], (2, 1, 0)).astype('float32')
        self.deform_reconmatrix = []
        self.control_idx = []
        self.vertex_dim = 9

        refmesh_V = np.transpose(data['ref_V']).astype('float32')
        refmesh_F = np.transpose(data['ref_F']).astype('int32') - 1
        self.refmesh_path = os.path.join('data', dataname + '.obj')
        self.mesh = pymesh.form_mesh(refmesh_V, refmesh_F)
        pymesh.save_mesh(self.refmesh_path, self.mesh, ascii=True)
        return

    def get_meshdata(self):
        print('transfer data')
        return self.feature, self.nb1, self.degree1, self.logrmin, self.logrmax, self.smin, self.smax, self.logr_avg, self.s_avg, \
               self.modelnum, self.pointnum1, self.maxdegree1, self.cotw1, self.vdiff, self.all_vertex, self.resultmin, self.resultmax


def convlayer_pooling(input_feature, input_dim, output_dim, nb, cotw, name='meshconv', training=True,
                      special_activation=True,
                      no_activation=False, bn=True):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))

        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)

        def compute_nb_feature(input_f):
            return tf.gather(input_f, nb) * cotw

        total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
        mean_nb_feature = tf.reduce_sum(total_nb_feature, axis=2)

        nb_weights = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32,
                                     tf.random_normal_initializer(stddev=0.02))
        nb_bias = tf.get_variable("nb_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2], [0]]) + nb_bias

        edge_weights = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32,
                                       tf.random_normal_initializer(stddev=0.02))
        edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        edge_feature = tf.tensordot(input_feature, edge_weights, [[2], [0]]) + edge_bias

        total_feature = edge_feature + nb_feature

        if not bn:
            fb = total_feature
        else:
            fb = batch_norm_wrapper(total_feature, is_training=training)

        if no_activation:
            fa = fb
        elif not special_activation:
            fa = leaky_relu(fb)
        else:
            fa = tf.tanh(fb)

        return fa


def newconvlayer_pooling(input_feature, input_dim, output_dim, nb_weights, edge_weights, nb, cotw,
                         name='meshconvpooling',
                         training=True, special_activation='fuck', no_activation=False, bn=True):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))

        padding_feature = tf.zeros([tf.shape(input_feature)[0], 1, input_dim], tf.float32)

        padded_input = tf.concat([padding_feature, input_feature], 1)
        def compute_nb_feature(input_f):
            return tf.gather(input_f, nb) * cotw

        total_nb_feature = tf.map_fn(compute_nb_feature, padded_input)
        mean_nb_feature = tf.reduce_sum(total_nb_feature, axis=2)

        nb_bias = tf.get_variable("nb_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        nb_feature = tf.tensordot(mean_nb_feature, nb_weights, [[2], [0]]) + nb_bias

        edge_bias = tf.get_variable("edge_bias", [output_dim], tf.float32, initializer=tf.constant_initializer(0.0))
        edge_feature = tf.tensordot(input_feature, edge_weights, [[2], [0]]) + edge_bias

        total_feature = edge_feature + nb_feature

        if not bn:
            fb = total_feature
        else:
            fb = batch_norm_wrapper(total_feature, is_training=training)

        if no_activation:
            fa = fb
            print('no activation')
        elif special_activation == 'sigmoid':
            fa = 2.0*tf.nn.sigmoid(fb)-1.0
            print('sigmoid')
        elif special_activation == 'l_relu':
            fa = leaky_relu(fb)
            print('l_relu')
        elif special_activation == 'softsign':
            fa = tf.nn.softsign(fb)
            print('softsign')
        else:
            fa = tf.nn.tanh(fb)
            print('tanh')

        return fa

def get_conv_weights(input_dim, output_dim, name='convweight'):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        n = tf.get_variable("nb_weights", [input_dim, output_dim], tf.float32,
                            tf.random_normal_initializer(stddev=0.02))
        e = tf.get_variable("edge_weights", [input_dim, output_dim], tf.float32,
                            tf.random_normal_initializer(stddev=0.02))
        return n, e


def loss_abs(a, b):
    return tf.reduce_mean(tf.abs(tf.subtract(a, b)))

def loss_mse(a, b):
    return tf.pow(tf.reduce_mean(tf.square(tf.subtract(a, b))), 0.5)

def load_lfd(path, result_max, result_min, dataname=''):
    data = h5py.File(path, mode = 'r')
    data = data[dataname]
    datalist = data.keys()

    dismat = np.transpose(data['dismat']).astype('float32')
    if 'lz_a' in datalist:
        metric_lz_a = data['lz_a']
        metric_lz_a = np.zeros_like(metric_lz_a).astype('float32')
        metric_lz_a = metric_lz_a.astype('float32')
    if 'lz_b' in datalist:
        metric_lz_b = data['lz_b']
        metric_lz_b = np.zeros_like(metric_lz_b).astype('float32')
        metric_lz_b = metric_lz_b.astype('float32')

    x = dismat
    x_min = x.min()
    x_min = x_min - 1e-6
    x_max = x.max()
    x_max = x_max + 1e-6

    x = (result_max - result_min) * (x - x_min) / (x_max - x_min) + result_min

    if 'lz_a' in datalist and 'lz_b' in datalist:
        a=1
    else:
        metric_lz_a = np.array([])
        metric_lz_b = np.array([])

    return x, x_min, x_max, metric_lz_a, metric_lz_b


def recover_data_old(dis, dismin, dismax, resultmin, resultmax):
    dis = (dismax - dismin) * (dis - resultmin) / (resultmax - resultmin) + dismin

    return dis

def recover_data(recover_feature, logrmin, logrmax, smin, smax, log_avg, s_avg, resultmin, resultmax, useS=True):
    logr = recover_feature[:, :, 0:3]

    logr = (logrmax - logrmin) * (logr - resultmin) / (resultmax - resultmin) + logrmin
    logr = logr + log_avg

    if useS:
        s = recover_feature[:, :, 3:9]
        s = (smax - smin) * (s - resultmin) / (resultmax - resultmin) + smin
        s = s + s_avg
        logr = np.concatenate((logr, s), axis=2)

    return logr


def leaky_relu(input_, alpha=0.2):
    return tf.maximum(input_, alpha * input_)


def linear_l2(input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float32,
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias, matrix


def linear(input_, input_size, output_size, name='Linear', stddev=0.02, bias_start=0.0):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        matrix = tf.get_variable("weights", [input_size, output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], tf.float32,
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

def linear1(input_, matrix, output_size, name='Linear', stddev=0.02, bias_start=0.0):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        bias = tf.get_variable("bias", [output_size], tf.float32,
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias


def batch_norm_wrapper(inputs, name='batch_norm', is_training=False, decay=0.9, epsilon=1e-5):
    with tf.variable_scope(name) as scope:
        scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True,
                                    initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True,
                                   initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False,
                                       initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False,
                                      initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

        if is_training == True:
            axis = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def spilt_dataset(num_a, num_b, percent_to_train, name="id.dat"):

    if os.path.isfile(name):
        id = pickle.load(open(name, 'rb'))
        id.show()
        Ia = id.Ia
        Ib = id.Ib
    else:
        Ia = np.arange(num_a)
        Ib = np.arange(num_b)
        Ia = random.sample(list(Ia), int(num_a * percent_to_train))
        Ib = random.sample(list(Ib), int(num_b * percent_to_train))
        id = Id(Ia, Ib)
        f = open(name, 'wb')
        pickle.dump(id, f, 0)
        f.close()
        id.show()

    Ia_C=list(set(np.arange(num_a)).difference(set(Ia)))
    Ib_C=list(set(np.arange(num_b)).difference(set(Ib)))

    return Ia, Ia_C, Ib, Ib_C



def printout(flog, data, epoch=0, interval = 50, write_to_file = True):
    if epoch % interval==0:
        print("data: ", data)
        flog.write(str((data+'\n')*write_to_file))

def argpaser2file(args, name='example.ini'):
    d = args.__dict__
    cfpar = configparser.ConfigParser()
    cfpar.optionxform = str
    cfpar['default'] = {}
    for key in sorted(d.keys()):
        cfpar['default'][str(key)]=str(d[key])
        print('%s = %s'%(key,d[key]))
    # print(cfpar['default'])

    with open(name, 'w') as configfile:
        cfpar.write(configfile)

def inifile2args(args, ininame='example.ini'):

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(ininame)
    defaults = config['default']
    result = dict(defaults)
    # print(result)
    # print('\n')
    # print(args)
    args1 = vars(args)
    # print(args1)

    args1.update({k: v for k, v in result.items() if v is not None})  # Update if v is not None

    #print(args1)
    args.__dict__.update(args1)

    #print(args)

    return args

def getFileName(path, postfix = '.ini'):
    filelist =[]
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        # os.path.splitext():
        if os.path.splitext(i)[1] == postfix:
            print("[{}] {}".format(f_list.index(i),i))
            filelist.append(i)

    return filelist


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def savemesh_pymesh(mesh, objpath, newv):
    new_mesh = pymesh.form_mesh(newv, mesh.faces)
    pymesh.save_mesh(objpath, new_mesh, ascii=True)


def gaussian(batch_size, n_dim, mean=0, var=1, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2:
            raise Exception("n_dim must be 2.")

        def sample(n_labels):
            x, y = np.random.normal(mean, var, (2,))
            angle = np.angle((x - mean) + 1j * (y - mean), deg=True)

            label = (int(n_labels * angle)) // 360

            if label < 0:
                label += n_labels

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size, 1), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range(int(n_dim / 2)):
                a_sample, a_label = sample(n_labels)
                z[batch, zi * 2:zi * 2 + 2] = a_sample
                z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
        return z

