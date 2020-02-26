# coding: utf-8
import os
import random
import time
import pickle
import tensorflow.contrib.slim as slim

from utils import *
from feature2vertex import *

test_vae = False
test_gan = False
test_metric = False
tb = False
test = False


class convMESH():
    VAE = 'VAE'
    METRIC = 'METRIC'
    GAN = 'GAN'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    def __init__(self, datainfo):
        self.logfolder = datainfo.output_dir
        if not os.path.isdir(self.logfolder):
            os.mkdir(self.logfolder)

        self.start = 'VAE'
        self.start_step_vae = 0
        self.start_step_metric = 0
        self.start_step_gan = 0

        self.n_epoch_Vae = datainfo.n_epoch_Vae
        self.n_epoch_Metric_1 = datainfo.n_epoch_Metric_1
        self.n_epoch_Gan = datainfo.n_epoch_Gan

        self.batch_size = 128
        self.hidden_dim = datainfo.hidden_dim
        self.lr = datainfo.speed
        self.useS = datainfo.useS
        self.use_tanh = datainfo.use_tanh
        self.use_sigmoid = datainfo.use_sigmoid
        self.sp_activ = datainfo.sp
        self.layer = datainfo.layer

        self.train_id_a, self.valid_id_a, self.train_id_b, self.valid_id_b = datainfo.train_id_a, datainfo.valid_id_a, datainfo.train_id_b, datainfo.valid_id_b

        self.feature_a, self.neighbour1_a, self.degree1_a, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a, self.s_avg_a, self.modelnum_a, \
        self.pointnum1_a, self.maxdegree1_a, self.cotw1_a, self.vdiff_a, self.all_vertex_a, self.resultmin_a, self.resultmax_a = datainfo.meshdata_a.get_meshdata()

        self.feature_b, self.neighbour1_b, self.degree1_b, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b, self.s_avg_b, self.modelnum_b, \
        self.pointnum1_b, self.maxdegree1_b, self.cotw1_b, self.vdiff_b, self.all_vertex_b, self.resultmin_b, self.resultmax_b = datainfo.meshdata_b.get_meshdata()

        self.lf_matrix, self.lf_matrix_min, self.lf_matrix_max, self.metric_lz_a, self.matric_lz_b = \
            datainfo.lf_matrix, datainfo.lf_matrix_min, datainfo.lf_matrix_max, datainfo.metric_lz_a, datainfo.matric_lz_b

        self.dataset_name_a = datainfo.featurefile_a.split('.')[0]
        self.pointnum1_a = self.pointnum1_a
        self.maxdegree1_a = self.maxdegree1_a
        self.model_num_a = self.modelnum_a
        self.lambda1_a = 10.0
        self.lambda2_a = 40.0

        self.dataset_name_b = datainfo.featurefile_b.split('.')[0]
        self.pointnum1_b = self.pointnum1_b
        self.maxdegree1_b = self.maxdegree1_b
        self.model_num_b = self.modelnum_b
        self.lambda1_b = 10.0
        self.lambda2_b = 40.0

        self.G_mapping = datainfo.G_mapping

        self.log_dir = self.logfolder
        self.vae_ablity = datainfo.vae_ablity

        self.checkpoint_dir = self.logfolder

        if not self.useS:
            self.vertex_dim = 3
            self.finaldim = 3
            datainfo.meshdata_a.vertex_dim = 3
            datainfo.meshdata_b.vertex_dim = 3
        else:
            self.vertex_dim = 9
            self.finaldim = 9

        self.f2v_a = Feature2Vertex(datainfo.meshdata_a, name='mesh_a')

        self.f2v_b = Feature2Vertex(datainfo.meshdata_b, name='mash_b')

        self.lf_dis = tf.placeholder(tf.float32, [None, 1], name='lf_dis')
        self.inputs_a = tf.placeholder(tf.float32, [None, self.pointnum1_a, self.vertex_dim], name='a/input_mesh')
        self.inputs_b = tf.placeholder(tf.float32, [None, self.pointnum1_b, self.vertex_dim], name='b/input_mesh')
        self.random_a = tf.placeholder(tf.float32, [None, self.hidden_dim], name='a/random_samples')
        self.random_b = tf.placeholder(tf.float32, [None, self.hidden_dim], name='b/random_samples')

        self.nb1_a = tf.constant(self.neighbour1_a, dtype='int32', shape=[self.pointnum1_a, self.maxdegree1_a],
                                 name='a/nb_relation1')

        self.degrees1_a = tf.constant(self.degree1_a, dtype='float32', shape=[self.pointnum1_a, 1], name='a/degrees1')

        self.cw1_a = tf.constant(self.cotw1_a, dtype='float32', shape=[self.pointnum1_a, self.maxdegree1_a, 1],
                                 name='a/cw1')

        self.nb2_a, self.degrees2_a = self.nb1_a, self.degrees1_a

        self.pointnum2_a, self.cw2_a = self.pointnum1_a, self.cw1_a

        self.object_stddev_a = tf.constant(np.concatenate((np.array([1, 1]).astype('float32'), 1
                                                           * np.ones(self.hidden_dim - 2).astype('float32'))))

        if self.layer >= 1:
            self.vae_n1_a, self.vae_e1_a = get_conv_weights(self.vertex_dim, self.vertex_dim, name='encoder/a/convw1')
            print("num layer|%d" % (self.layer))
        if self.layer >= 2:
            self.vae_n2_a, self.vae_e2_a = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/a/convw2')
            print("num layer|%d" % (self.layer))
        if self.layer >= 3:
            self.vae_n3_a, self.vae_e3_a = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/a/convw3')
            print("num layer|%d" % (self.layer))
        if self.layer >= 4:
            self.vae_n4_a, self.vae_e4_a = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/a/convw4')
            print("num layer|%d" % (self.layer))
        if self.layer >= 5:
            self.vae_n5_a, self.vae_e5_a = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/a/convw5')
            print("num layer|%d" % (self.layer))
        self.vae_meanpara_a = tf.get_variable("encoder/a/mean_weights",
                                              [self.pointnum2_a * self.finaldim, self.hidden_dim], tf.float32,
                                              tf.random_normal_initializer(stddev=0.02))

        self.vae_stdpara_a = tf.get_variable("encoder/a/std_weights",
                                             [self.pointnum2_a * self.finaldim, self.hidden_dim], tf.float32,
                                             tf.random_normal_initializer(stddev=0.02))

        self.z_mean_a, self.z_stddev_a = self.encoder_a(self.inputs_a)
        self.guessed_z_a = self.z_mean_a + self.z_stddev_a * self.random_a

        self.z_mean_test_a, self.z_stddev_test_a = self.encoder_a(self.inputs_a, train=False)
        self.guessed_z_test_a = self.z_mean_test_a + self.z_stddev_test_a * self.random_a

        self.generated_mesh_train_a = self.decoder_a(self.guessed_z_a, train=True)
        self.generated_mesh_train_a = tf.clip_by_value(self.generated_mesh_train_a, datainfo.resultmin + 1e-8,
                                                       datainfo.resultmax - 1e-8)
        self.generated_mesh_test_a = self.decoder_a(self.guessed_z_test_a, train=False)
        self.test_mesh_a = self.decoder_a(self.random_a, train=False)

        marginal_likelihood_a = tf.reduce_sum(tf.pow(self.inputs_a - self.generated_mesh_train_a, 2.0),
                                              [1, 2])
        KL_divergence_a = 0.5 * tf.reduce_sum(
            tf.square(self.z_mean_a) + tf.square(self.z_stddev_a / self.object_stddev_a) - tf.log(
                1e-8 + tf.square(self.z_stddev_a / self.object_stddev_a)) - 1, 1)

        self.neg_loglikelihood_a = self.lambda2_a * tf.reduce_mean(marginal_likelihood_a)
        self.KL_divergence_a = self.lambda1_a * tf.reduce_mean(KL_divergence_a)

        ELBO_a = - self.neg_loglikelihood_a - self.KL_divergence_a
        if self.layer >= 1:
            self.r2_a = tf.nn.l2_loss(self.vae_e1_a) + tf.nn.l2_loss(self.vae_n1_a)
            print("num layer|%d" % (self.layer))
        if self.layer >= 2:
            self.r2_a = self.r2_a + tf.nn.l2_loss(self.vae_e2_a) + tf.nn.l2_loss(self.vae_n2_a)
            print("num layer|%d" % (self.layer))
        if self.layer >= 3:
            self.r2_a = self.r2_a + tf.nn.l2_loss(self.vae_e3_a) + tf.nn.l2_loss(self.vae_n3_a)
            print("num layer|%d" % (self.layer))
        if self.layer >= 4:
            self.r2_a = self.r2_a + tf.nn.l2_loss(self.vae_e4_a) + tf.nn.l2_loss(self.vae_n4_a)
            print("num layer|%d" % (self.layer))
        if self.layer >= 5:
            self.r2_a = self.r2_a + tf.nn.l2_loss(self.vae_e5_a) + tf.nn.l2_loss(self.vae_n5_a)
            print("num layer|%d" % (self.layer))

        self.r2_a = self.r2_a + tf.nn.l2_loss(self.vae_stdpara_a) + tf.nn.l2_loss(
            self.vae_meanpara_a)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='encoder/a') + tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope='decoder/a')
        self.r2_a += sum(reg_losses)


        self.nb1_b = tf.constant(self.neighbour1_b, dtype='int32', shape=[self.pointnum1_b, self.maxdegree1_b],
                                 name='b/nb_relation1')
        self.degrees1_b = tf.constant(self.degree1_b, dtype='float32', shape=[self.pointnum1_b, 1], name='b/degrees1')
        self.cw1_b = tf.constant(self.cotw1_b, dtype='float32', shape=[self.pointnum1_b, self.maxdegree1_b, 1],
                                 name='b/cw1')

        self.nb2_b, self.degrees2_b = self.nb1_b, self.degrees1_b
        self.pointnum2_b, self.cw2_b = self.pointnum1_b, self.cw1_b

        self.object_stddev_b = tf.constant(
            np.concatenate((np.array([1, 1]).astype('float32'), 1 * np.ones(self.hidden_dim - 2).astype('float32'))))

        if self.layer >= 1:
            self.vae_n1_b, self.vae_e1_b = get_conv_weights(self.vertex_dim, self.vertex_dim, name='encoder/b/convw1')
            print("num layer|%d" % (self.layer))
        if self.layer >= 2:
            self.vae_n2_b, self.vae_e2_b = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/b/convw2')
            print("num layer|%d" % (self.layer))
        if self.layer >= 3:
            self.vae_n3_b, self.vae_e3_b = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/b/convw3')
            print("num layer|%d" % (self.layer))
        if self.layer >= 4:
            self.vae_n4_b, self.vae_e4_b = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/b/convw4')
            print("num layer|%d" % (self.layer))
        if self.layer >= 5:
            self.vae_n5_b, self.vae_e5_b = get_conv_weights(self.vertex_dim, self.finaldim, name='encoder/b/convw5')
            print("num layer|%d" % (self.layer))
        self.vae_meanpara_b = tf.get_variable("encoder/b/mean_weights",
                                              [self.pointnum2_b * self.finaldim, self.hidden_dim], tf.float32,
                                              tf.contrib.layers.xavier_initializer())
        self.vae_stdpara_b = tf.get_variable("encoder/b/std_weights",
                                             [self.pointnum2_b * self.finaldim, self.hidden_dim], tf.float32,
                                             tf.contrib.layers.xavier_initializer())

        self.z_mean_b, self.z_stddev_b = self.encoder_b(self.inputs_b)
        self.guessed_z_b = self.z_mean_b + self.z_stddev_b * self.random_b

        self.z_mean_test_b, self.z_stddev_test_b = self.encoder_b(self.inputs_b, train=False)
        self.guessed_z_test_b = self.z_mean_test_b + self.z_stddev_test_b * self.random_b

        self.generated_mesh_train_b = self.decoder_b(self.guessed_z_b, train=True)
        self.generated_mesh_train_b = tf.clip_by_value(self.generated_mesh_train_b, datainfo.resultmin + 1e-8,
                                                       datainfo.resultmax - 1e-8)
        self.generated_mesh_test_b = self.decoder_b(self.guessed_z_test_b, train=False)
        self.test_mesh_b = self.decoder_b(self.random_b, train=False)

        marginal_likelihood_b = tf.reduce_sum(tf.pow(self.inputs_b - self.generated_mesh_train_b, 2.0),
                                              [1, 2])
        KL_divergence_b = 0.5 * tf.reduce_sum(
            tf.square(self.z_mean_b) + tf.square(self.z_stddev_b / self.object_stddev_b) - tf.log(
                1e-8 + tf.square(self.z_stddev_b / self.object_stddev_b)) - 1, 1)

        self.neg_loglikelihood_b = self.lambda2_b * tf.reduce_mean(marginal_likelihood_b)
        self.KL_divergence_b = self.lambda1_b * tf.reduce_mean(KL_divergence_b)

        ELBO_b = - self.neg_loglikelihood_b - self.KL_divergence_b
        if self.layer >= 1:
            self.r2_b = tf.nn.l2_loss(self.vae_e1_b) + tf.nn.l2_loss(self.vae_n1_b)
            print("num layer|%d" % (self.layer))
        if self.layer >= 2:
            self.r2_b = self.r2_b + tf.nn.l2_loss(self.vae_e2_b) + tf.nn.l2_loss(self.vae_n2_b)
            print("num layer|%d" % (self.layer))
        if self.layer >= 3:
            self.r2_b = self.r2_b + tf.nn.l2_loss(self.vae_e3_b) + tf.nn.l2_loss(self.vae_n3_b)
            print("num layer|%d" % (self.layer))
        if self.layer >= 4:
            self.r2_b = self.r2_b + tf.nn.l2_loss(self.vae_e4_b) + tf.nn.l2_loss(self.vae_n4_b)
            print("num layer|%d" % (self.layer))
        if self.layer >= 5:
            self.r2_b = self.r2_b + tf.nn.l2_loss(self.vae_e5_b) + tf.nn.l2_loss(self.vae_n5_b)
            print("num layer|%d" % (self.layer))

        self.r2_b = self.r2_b + tf.nn.l2_loss(self.vae_stdpara_b) + tf.nn.l2_loss(
            self.vae_meanpara_b)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='encoder/b') + tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES, scope='decoder/b')

        self.r2_b += sum(reg_losses)

        self.distance = self.metric_net(self.z_mean_test_a, self.z_mean_test_b)

        self.loss_metric_l2 = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='metric'))

        if not self.use_tanh:
            self.random_gen_latent_a, self.random_gen_stddev_a = self.gen_a(self.random_a)
            self.random_gen_latent_b, self.random_gen_stddev_b = self.gen_b(self.random_b)
            self.random_gen_feature_a = self.decoder_b(self.random_gen_latent_a, train=False)
            self.random_gen_feature_b = self.decoder_a(self.random_gen_latent_b, train=False)

            self.gen_latent_a, _ = self.gen_a(self.z_mean_test_a, reuse=True)
            self.gen_latent_b, _ = self.gen_b(self.z_mean_test_b, reuse=True)
            self.gen_feature_a = self.decoder_b(self.gen_latent_a, train=False)
            self.gen_feature_b = self.decoder_a(self.gen_latent_b, train=False)

            self.dis_real_a = self.dis_a(self.generated_mesh_test_b)
            self.dis_real_b = self.dis_b(self.generated_mesh_test_a)
            self.dis_fake_a = self.dis_a(self.random_gen_feature_a, reuse=True)
            self.dis_fake_b = self.dis_b(self.random_gen_feature_b, reuse=True)
            self.real_label = tf.multiply(tf.ones_like(self.dis_real_a), 1.00)
            self.fake_label = tf.add(tf.zeros_like(self.dis_fake_a), 0.00)

            self.cycle_z_a, _ = self.gen_b(self.gen_latent_a, reuse=True)
            self.cycle_z_b, _ = self.gen_a(self.gen_latent_b, reuse=True)
        else:
            self.random_gen_latent_a = self.gen_a(self.random_a)
            self.random_gen_latent_b = self.gen_b(self.random_b)
            self.random_gen_feature_a = self.decoder_b(self.random_gen_latent_a, train=False)
            self.random_gen_feature_b = self.decoder_a(self.random_gen_latent_b, train=False)

            self.gen_latent_a = self.gen_a(self.z_mean_test_a, reuse=True)
            self.gen_latent_b = self.gen_b(self.z_mean_test_b, reuse=True)
            self.gen_feature_a = self.decoder_b(self.gen_latent_a, train=False)
            self.gen_feature_b = self.decoder_a(self.gen_latent_b, train=False)

            self.dis_real_a = self.dis_a(self.gen_feature_a)
            self.dis_real_b = self.dis_b(self.gen_feature_b)
            self.dis_fake_a = self.dis_a(self.random_gen_feature_a, reuse=True)
            self.dis_fake_b = self.dis_b(self.random_gen_feature_b, reuse=True)
            self.real_label = tf.multiply(tf.ones_like(self.dis_real_a), 1.00)
            self.fake_label = tf.add(tf.zeros_like(self.dis_fake_a), 0.00)

            self.cycle_z_a = self.gen_b(self.gen_latent_a, reuse=True)
            self.cycle_z_b = self.gen_a(self.gen_latent_b, reuse=True)


        self.distance_test = self.metric_net(self.random_a, self.random_b, reuse=True)

        self.decoder_x = self.decoder_a(self.z_mean_test_a, train=False)
        self.decoder_gx = self.decoder_b(self.gen_latent_a, train=False)
        self.decoder_fgx = self.decoder_a(self.cycle_z_a, train=False)

        self.decoder_y = self.decoder_b(self.z_mean_test_b, train=False)
        self.decoder_fy = self.decoder_a(self.gen_latent_b, train=False)
        self.decoder_gfy = self.decoder_b(self.cycle_z_b, train=False)

        self.loss_mapping = tf.reduce_mean(self.metric_net(self.random_a, self.random_gen_latent_a, training=False) \
                                           + self.metric_net(self.random_gen_latent_b, self.random_b, training=False))
        self.loss_mapping = self.loss_mapping * self.G_mapping + 10.0*(loss_mse(self.gen_latent_a, self.z_mean_test_b) + loss_mse(self.gen_latent_b, self.z_mean_test_a))
        self.loss_KL = tf.constant(0.0, dtype='float32')

        if not self.use_tanh:
            loss_gen_KL_a = 0.5 * tf.reduce_sum(
                tf.square(self.random_gen_latent_a) + tf.square(
                    self.random_gen_stddev_a / self.object_stddev_a) - tf.log(
                    1e-8 + tf.square(self.random_gen_stddev_a / self.object_stddev_a)) - 1, 1)
            loss_gen_KL_b = 0.5 * tf.reduce_sum(
                tf.square(self.random_gen_latent_b) + tf.square(
                    self.random_gen_stddev_b / self.object_stddev_b) - tf.log(
                    1e-8 + tf.square(self.random_gen_stddev_b / self.object_stddev_b)) - 1, 1)
            self.loss_KL = 0.0 * tf.reduce_mean(loss_gen_KL_a + loss_gen_KL_b)
            self.loss_mapping = self.loss_mapping + self.loss_KL

        self.loss_cycle = loss_mse(self.z_mean_test_a, self.cycle_z_a) + loss_mse(self.z_mean_test_b, self.cycle_z_b)
        self.loss_cycle = self.loss_cycle * 20.0
        self.loss_d = loss_mse(self.dis_real_a, self.real_label) + loss_mse(self.dis_real_b, self.real_label) \
                      + loss_mse(self.dis_fake_a, self.fake_label) + loss_mse(self.dis_fake_b, self.fake_label)

        self.loss_d = self.loss_d * 00.0001
        self.loss_g = loss_mse(self.dis_fake_a, self.real_label) + loss_mse(self.dis_fake_b, self.real_label)
        self.loss_g = self.loss_g * 20.0

        self.loss_g_all = self.loss_mapping + self.loss_g + self.loss_cycle

        self.loss_metric_l2 = self.loss_metric_l2 * 0.01
        self.loss_metric_max = tf.reduce_max(tf.abs(self.lf_dis - self.distance))
        self.loss_metric = loss_mse(self.lf_dis, self.distance) * 1000 + self.loss_metric_l2
        self.loss_d_all = self.loss_d

        self.loss_vae_a = -ELBO_a + 0.1 * self.r2_a
        self.loss_vae_b = -ELBO_b + 0.1 * self.r2_b

        if tb:
            tf.summary.scalar('loss_g', self.loss_g)
            tf.summary.scalar('loss_cycle', self.loss_cycle)
            tf.summary.scalar('loss_mapping', self.loss_mapping)
            tf.summary.scalar('loss_d_all', self.loss_d_all)
            tf.summary.scalar('loss_g_all', self.loss_g_all)
            tf.summary.scalar('loss_metric', self.loss_metric)
            tf.summary.scalar('loss_metric_l2', self.loss_metric_l2)

            tf.summary.scalar('loss_vae_a', self.loss_vae_a)
            tf.summary.scalar('loss_vae_b', self.loss_vae_b)

            tf.summary.scalar("nll_a", self.neg_loglikelihood_a)
            tf.summary.scalar("kl_a", self.KL_divergence_a)
            tf.summary.scalar("L2_loss_a", self.r2_a)
            tf.summary.scalar("nll_b", self.neg_loglikelihood_b)
            tf.summary.scalar("kl_b", self.KL_divergence_b)
            tf.summary.scalar("L2_loss_b", self.r2_b)

        self.decay_rate = 0.8
        self.decay_step = 1000
        global_step_vae_a = tf.Variable(0, trainable=False, name='global_step_vae_a')
        learning_rate_vae_a = tf.train.exponential_decay(self.lr, global_step_vae_a, self.decay_step, self.decay_rate,
                                                         staircase=True)
        learning_rate_vae_a = tf.maximum(learning_rate_vae_a, 0.0000001)

        global_step_vae_b = tf.Variable(0, trainable=False, name='global_step_vae_b')
        learning_rate_vae_b = tf.train.exponential_decay(self.lr, global_step_vae_b, self.decay_step, self.decay_rate,
                                                         staircase=True)
        learning_rate_vae_b = tf.maximum(learning_rate_vae_b, 0.0000001)

        global_step_g = tf.Variable(0, trainable=False, name='global_step_g')
        learning_rate_g = tf.train.exponential_decay(self.lr, global_step_g, self.decay_step * 2, self.decay_rate,
                                                     staircase=True)
        learning_rate_g = tf.maximum(learning_rate_g, 0.001)

        global_step_d = tf.Variable(0, trainable=False, name='global_step_d')
        learning_rate_d = tf.train.exponential_decay(self.lr * 0.5, global_step_d, self.decay_step, self.decay_rate,
                                                     staircase=True)
        learning_rate_d = tf.maximum(learning_rate_d, 0.0000001)

        global_step_metric_1 = tf.Variable(0, trainable=False, name='global_step_metric_1')
        learning_rate_metric_1 = tf.train.exponential_decay(self.lr, global_step_metric_1, self.decay_step, 0.9,
                                                            staircase=True)
        learning_rate_metric_1 = tf.maximum(learning_rate_metric_1, 0.0000001)

        self.optimizer_vae_a = tf.train.AdamOptimizer(learning_rate_vae_a, name='encoder/a')
        self.optimizer_vae_b = tf.train.AdamOptimizer(learning_rate_vae_b, name='encoder/b')

        self.optimizer_g = tf.train.AdamOptimizer(learning_rate_g, name='gen')
        self.optimizer_d = tf.train.AdamOptimizer(learning_rate_d, name='dis')
        self.optimizer_metric_1 = tf.train.AdamOptimizer(learning_rate_metric_1, name='metric1')


        variables_encoder_a = slim.get_variables(scope="encoder/a")
        variables_encoder_b = slim.get_variables(scope="encoder/b")
        variables_decoder_a = slim.get_variables(scope="decoder/a")
        variables_decoder_b = slim.get_variables(scope="decoder/b")

        variables_metric = slim.get_variables(scope="metric")
        variables_g = slim.get_variables(scope="gen")
        variables_d = slim.get_variables(scope="dis")

        train_variables_vae_a = []
        train_variables_vae_b = []
        train_variables_vae_all = []
        variables_vae_all = []
        variables_vae_a = []
        variables_vae_b = []

        train_variables_g = []
        train_variables_d = []
        train_variables_metric = []
        varaibles_gan_all = []
        variables_vae_metric = []

        for v in variables_encoder_a:
            variables_vae_metric.append(v)
            variables_vae_a.append(v)
            variables_vae_all.append(v)
            if v in tf.trainable_variables():
                train_variables_vae_a.append(v)
                train_variables_vae_all.append(v)
        for v in variables_decoder_a:
            variables_vae_metric.append(v)
            variables_vae_a.append(v)
            variables_vae_all.append(v)
            if v in tf.trainable_variables():
                train_variables_vae_a.append(v)
                train_variables_vae_all.append(v)
        for v in variables_encoder_b:
            variables_vae_metric.append(v)
            variables_vae_b.append(v)
            variables_vae_all.append(v)
            if v in tf.trainable_variables():
                train_variables_vae_b.append(v)
                train_variables_vae_all.append(v)
        for v in variables_decoder_b:
            variables_vae_metric.append(v)
            variables_vae_b.append(v)
            variables_vae_all.append(v)
            if v in tf.trainable_variables():
                train_variables_vae_b.append(v)
                train_variables_vae_all.append(v)

        for v in variables_g:
            varaibles_gan_all.append(v)
            if v in tf.trainable_variables():
                train_variables_g.append(v)
        for v in variables_d:
            varaibles_gan_all.append(v)
            if v in tf.trainable_variables():
                train_variables_d.append(v)
        for v in variables_metric:
            variables_vae_metric.append(v)
            if v in tf.trainable_variables():
                train_variables_metric.append(v)
        print(train_variables_g)
        self.saver = tf.train.Saver(max_to_keep=None)

        self.train_op_vae_a = tf.contrib.training.create_train_op(self.loss_vae_a, self.optimizer_vae_a,
                                                                  global_step=global_step_vae_a,
                                                                  variables_to_train=train_variables_vae_a,
                                                                  summarize_gradients=False)
        self.train_op_vae_b = tf.contrib.training.create_train_op(self.loss_vae_b, self.optimizer_vae_b,
                                                                  global_step=global_step_vae_b,
                                                                  variables_to_train=train_variables_vae_b,
                                                                  summarize_gradients=False)

        self.train_op_g = tf.contrib.training.create_train_op(self.loss_g_all, self.optimizer_g,
                                                              global_step=global_step_g,
                                                              variables_to_train=train_variables_g,
                                                              summarize_gradients=False)
        self.train_op_d = tf.contrib.training.create_train_op(self.loss_d_all, self.optimizer_d,
                                                              global_step=global_step_d,
                                                              variables_to_train=train_variables_d,
                                                              summarize_gradients=False)
        self.train_op_metric_1 = tf.contrib.training.create_train_op(self.loss_metric, self.optimizer_metric_1,
                                                                     global_step=global_step_metric_1,
                                                                     variables_to_train=train_variables_metric,
                                                                     summarize_gradients=False)

        self.checkpoint_dir = os.path.join(self.logfolder, 'checkpoint')
        timecurrent = time.strftime('%m%d%H%M%S', time.localtime(time.time()))
        if os.path.exists(self.logfolder + '/log.txt'):
            self.file = open(self.logfolder + '/log.txt', 'a')
        else:
            self.file = open(self.logfolder + '/log.txt', 'w')

        if tb:
            self.merge_summary = tf.summary.merge_all()

    def metric_net(self, A, B, training=True, reuse=False):
        with tf.variable_scope("metric") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not training or reuse:
                training = False
                reuse = True
                scope.reuse_variables()

            h1, _ = linear_l2(tf.concat([A, B], axis=1), self.hidden_dim * 2, 2048, 'h1')
            h1bn = batch_norm_wrapper(h1, name='h1bn', is_training=training, decay=0.9)
            h1a = leaky_relu(h1bn)

            h2, _ = linear_l2(h1a, 2048, 1024, 'h2')
            h2bn = batch_norm_wrapper(h2, name='h2bn', is_training=training, decay=0.9)
            h2a = leaky_relu(h2bn)

            h3, _ = linear_l2(h2a, 1024, 512, 'h3')
            h3bn = batch_norm_wrapper(h3, name='h3bn', is_training=training, decay=0.9)
            h3a = leaky_relu(h3bn)

            h4, _ = linear_l2(h3a, 512, 256, 'h4')
            h4bn = batch_norm_wrapper(h4, name='h4bn', is_training=training, decay=0.9)
            h4a = leaky_relu(h4bn)

            distance, _ = linear_l2(h4a, 256, 1, 'distance')
            return tf.square(distance)

    def encoder_a(self, input_feature, train=True, reuse=False):
        with tf.variable_scope("encoder/a") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()
            bnbn = True
            if self.layer >= 1:
                print("num layer|%d" % (self.layer))
                if self.layer == 1:
                    bnbn = False
                conv1 = newconvlayer_pooling(input_feature, self.vertex_dim, self.vertex_dim, self.vae_n1_a,
                                             self.vae_e1_a, self.nb1_a,
                                             self.cw1_a, name='conv1', training=train, bn=bnbn,
                                             special_activation=self.sp_activ)
            if self.layer >= 2:
                print("num layer|%d" % (self.layer))
                if self.layer == 2:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n2_a, self.vae_e2_a,
                                             self.nb1_a, self.cw1_a,
                                             name='conv2', training=train, bn=bnbn, special_activation=self.sp_activ)
                print('self.layer')
            if self.layer >= 3:
                print("num layer|%d" % (self.layer))
                if self.layer == 3:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n3_a, self.vae_e3_a,
                                             self.nb1_a, self.cw1_a,
                                             name='conv3', training=train, bn=bnbn, special_activation=self.sp_activ)
            if self.layer >= 4:
                print("num layer|%d" % (self.layer))
                if self.layer == 4:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n4_a, self.vae_e4_a,
                                             self.nb1_a, self.cw1_a,
                                             name='conv4', training=train, bn=bnbn, special_activation=self.sp_activ)
            if self.layer >= 5:
                print("num layer|%d" % (self.layer))
                if self.layer == 5:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n5_a, self.vae_e5_a,
                                             self.nb1_a, self.cw1_a,
                                             name='conv5', training=train, bn=bnbn, special_activation=self.sp_activ)

            l0 = tf.reshape(conv1, [tf.shape(conv1)[0], self.pointnum2_a * self.finaldim])
            mean = tf.nn.tanh(linear1(l0, self.vae_meanpara_a, self.hidden_dim, 'mean'))
            stddev = linear1(l0, self.vae_stdpara_a, self.hidden_dim, 'stddev')
            stddev = tf.sqrt(2 * tf.nn.sigmoid(stddev))

            return mean, stddev

    def decoder_a(self, latent_tensor, train=True, reuse=False):
        with tf.variable_scope("decoder/a") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            l1 = linear1(latent_tensor, tf.transpose(self.vae_meanpara_a), self.pointnum2_a * self.finaldim, 'mean')
            conv1 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum2_a, self.finaldim])

            if self.layer == 1:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_a),
                                             tf.transpose(self.vae_e1_a),
                                             self.nb2_a, self.cw2_a, name='conv2', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 2:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n2_a),
                                             tf.transpose(self.vae_e2_a),
                                             self.nb2_a, self.cw2_a, name='conv3', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_a),
                                             tf.transpose(self.vae_e1_a),
                                             self.nb2_a, self.cw2_a, name='conv4', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 3:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n3_a),
                                             tf.transpose(self.vae_e3_a),
                                             self.nb2_a, self.cw2_a, name='conv4', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n2_a),
                                             tf.transpose(self.vae_e2_a),
                                             self.nb2_a, self.cw2_a, name='conv5', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_a),
                                             tf.transpose(self.vae_e1_a),
                                             self.nb2_a, self.cw2_a, name='conv6', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 4:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n4_a),
                                             tf.transpose(self.vae_e4_a),
                                             self.nb2_a, self.cw2_a, name='conv5', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n3_a),
                                             tf.transpose(self.vae_e3_a),
                                             self.nb2_a, self.cw2_a, name='conv6', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n2_a),
                                             tf.transpose(self.vae_e2_a),
                                             self.nb2_a, self.cw2_a, name='conv7', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_a),
                                             tf.transpose(self.vae_e1_a),
                                             self.nb2_a, self.cw2_a, name='conv8', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 5:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n5_a),
                                             tf.transpose(self.vae_e5_a),
                                             self.nb2_a, self.cw2_a, name='conv6', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n4_a),
                                             tf.transpose(self.vae_e4_a),
                                             self.nb2_a, self.cw2_a, name='conv7', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n3_a),
                                             tf.transpose(self.vae_e3_a),
                                             self.nb2_a, self.cw2_a, name='conv8', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n2_a),
                                             tf.transpose(self.vae_e2_a),
                                             self.nb2_a, self.cw2_a, name='conv9', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_a),
                                             tf.transpose(self.vae_e1_a),
                                             self.nb2_a, self.cw2_a, name='conv10', training=train, bn=False,
                                             special_activation=self.sp_activ)

        return conv1

    def gen_a(self, latent_tensor, train=True, reuse=False):
        with tf.variable_scope("gen/a") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            h1 = linear(latent_tensor, self.hidden_dim, 256, 'h1')
            h1bn = batch_norm_wrapper(h1, name='h1bn', is_training=train, decay=0.9)
            h1a = leaky_relu(h1bn)

            h2 = linear(h1a, 256, 512, 'h2')
            h2bn = batch_norm_wrapper(h2, name='h2bn', is_training=train, decay=0.9)
            h2a = leaky_relu(h2bn)

            h3 = linear(h2a, 512, 1024, 'h3')
            h3bn = batch_norm_wrapper(h3, name='h3bn', is_training=train, decay=0.9)
            h3a = leaky_relu(h3bn)

            h4 = linear(h3a, 1024, 512, 'h4')
            h4bn = batch_norm_wrapper(h4, name='h4bn', is_training=train, decay=0.9)
            h4a = leaky_relu(h4bn)
            h5a = h4a

            if not self.use_tanh:
                out_mean = linear(h5a, 512, self.hidden_dim, name='out_mean')
                out_std = tf.sqrt(2 * tf.nn.sigmoid(linear(h5a, 512, self.hidden_dim, name='out_std')))
                return out_mean, out_std
            else:
                out_mean = linear(h5a, 512, self.hidden_dim, 'out_mean')# + latent_tensor
                return out_mean#tf.nn.tanh(out_mean)

    def dis_a(self, input_feature, train=True, reuse=False):
        with tf.variable_scope("dis/a") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            conv1 = convlayer_pooling(input_feature, self.vertex_dim, self.vertex_dim, self.nb1_b, self.cw1_b,
                                      name='conv1', training=train)

            conv1 = convlayer_pooling(conv1, self.vertex_dim, self.finaldim, self.nb2_b, self.cw2_b, name='conv3',
                                      training=train, bn=False)
            l0 = tf.reshape(conv1, [tf.shape(conv1)[0], self.pointnum2_b * self.finaldim])

            if self.use_sigmoid == '2s':
                l0 = linear(l0, self.pointnum2_b * self.finaldim, 2048, 'dense1')
                return tf.nn.sigmoid(linear(l0, 2048, 1, 'dense2'))
            elif self.use_sigmoid == '1s':
                return tf.nn.sigmoid(linear(l0, self.pointnum2_b * self.finaldim, 1, 'dense1'))
            elif self.use_sigmoid == '2':
                l0 = linear(l0, self.pointnum2_b * self.finaldim, 2048, 'dense1')
                return linear(l0, 2048, 1, 'dense2')
            else:
                return linear(l0, self.pointnum2_b * self.finaldim, 512, 'dense1')

    def encoder_b(self, input_feature, train=True, reuse=False):
        with tf.variable_scope("encoder/b") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()
            bnbn = True
            if self.layer >= 1:
                print("num layer|%d" % (self.layer))
                if self.layer == 1:
                    bnbn = False
                conv1 = newconvlayer_pooling(input_feature, self.vertex_dim, self.vertex_dim, self.vae_n1_b,
                                             self.vae_e1_b, self.nb1_b,
                                             self.cw1_b, name='conv1', training=train, bn=bnbn,
                                             special_activation=self.sp_activ)
            if self.layer >= 2:
                print("num layer|%d" % (self.layer))
                if self.layer == 2:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n2_b, self.vae_e2_b,
                                             self.nb1_b, self.cw1_b,
                                             name='conv2', training=train, bn=bnbn, special_activation=self.sp_activ)
            if self.layer >= 3:
                print("num layer|%d" % (self.layer))
                if self.layer == 3:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n3_b, self.vae_e3_b,
                                             self.nb1_b, self.cw1_b,
                                             name='conv3', training=train, bn=bnbn, special_activation=self.sp_activ)
            if self.layer >= 4:
                print("num layer|%d" % (self.layer))
                if self.layer == 4:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n4_b, self.vae_e4_b,
                                             self.nb1_b, self.cw1_b,
                                             name='conv4', training=train, bn=bnbn, special_activation=self.sp_activ)
            if self.layer >= 5:
                print("num layer|%d" % (self.layer))
                if self.layer == 5:
                    bnbn = False
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, self.vae_n5_b, self.vae_e5_b,
                                             self.nb1_b, self.cw1_b,
                                             name='conv5', training=train, bn=bnbn, special_activation=self.sp_activ)

            l0 = tf.reshape(conv1, [tf.shape(conv1)[0], self.pointnum2_b * self.finaldim])
            mean = tf.nn.tanh(linear1(l0, self.vae_meanpara_b, self.hidden_dim, 'mean'))
            stddev = linear1(l0, self.vae_stdpara_b, self.hidden_dim, 'stddev')
            stddev = tf.sqrt(2 * tf.nn.sigmoid(stddev))

            return mean, stddev

    def decoder_b(self, latent_tensor, train=True, reuse=False):
        with tf.variable_scope("decoder/b") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            l1 = linear1(latent_tensor, tf.transpose(self.vae_meanpara_b), self.pointnum2_b * self.finaldim, 'mean')

            conv1 = tf.reshape(l1, [tf.shape(l1)[0], self.pointnum2_b, self.finaldim])
            if self.layer == 1:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n1_b),
                                             tf.transpose(self.vae_e1_b),
                                             self.nb2_b, self.cw2_b, name='conv2', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 2:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n2_b),
                                             tf.transpose(self.vae_e2_b),
                                             self.nb2_b, self.cw2_b, name='conv3', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n1_b),
                                             tf.transpose(self.vae_e1_b),
                                             self.nb2_b, self.cw2_b, name='conv4', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 3:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n3_b),
                                             tf.transpose(self.vae_e3_b),
                                             self.nb2_b, self.cw2_b, name='conv4', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n2_b),
                                             tf.transpose(self.vae_e2_b),
                                             self.nb2_b, self.cw2_b, name='conv5', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n1_b),
                                             tf.transpose(self.vae_e1_b),
                                             self.nb2_b, self.cw2_b, name='conv6', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 4:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n4_b),
                                             tf.transpose(self.vae_e4_b),
                                             self.nb2_b, self.cw2_b, name='conv5', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n3_b),
                                             tf.transpose(self.vae_e3_b),
                                             self.nb2_b, self.cw2_b, name='conv6', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n2_b),
                                             tf.transpose(self.vae_e2_b),
                                             self.nb2_b, self.cw2_b, name='conv7', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n1_b),
                                             tf.transpose(self.vae_e1_b),
                                             self.nb2_b, self.cw2_b, name='conv8', training=train, bn=False,
                                             special_activation=self.sp_activ)
            if self.layer == 5:
                print("num layer|%d" % (self.layer))
                conv1 = newconvlayer_pooling(conv1, self.finaldim, self.vertex_dim, tf.transpose(self.vae_n5_b),
                                             tf.transpose(self.vae_e5_b),
                                             self.nb2_b, self.cw2_b, name='conv6', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n4_b),
                                             tf.transpose(self.vae_e4_b),
                                             self.nb2_b, self.cw2_b, name='conv7', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n3_b),
                                             tf.transpose(self.vae_e3_b),
                                             self.nb2_b, self.cw2_b, name='conv8', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n2_b),
                                             tf.transpose(self.vae_e2_b),
                                             self.nb2_b, self.cw2_b, name='conv9', training=train,
                                             special_activation=self.sp_activ)
                conv1 = newconvlayer_pooling(conv1, self.vertex_dim, self.vertex_dim, tf.transpose(self.vae_n1_b),
                                             tf.transpose(self.vae_e1_b),
                                             self.nb2_b, self.cw2_b, name='conv10', training=train, bn=False,
                                             special_activation=self.sp_activ)
        return conv1

    def gen_b(self, latent_tensor, train=True, reuse=False):
        with tf.variable_scope("gen/b") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            h1 = linear(latent_tensor, self.hidden_dim, 256, 'h1')
            h1bn = batch_norm_wrapper(h1, name='h1bn', is_training=train, decay=0.9)
            h1a = leaky_relu(h1bn)

            h2 = linear(h1a, 256, 512, 'h2')
            h2bn = batch_norm_wrapper(h2, name='h2bn', is_training=train, decay=0.9)
            h2a = leaky_relu(h2bn)

            h3 = linear(h2a, 512, 1024, 'h3')
            h3bn = batch_norm_wrapper(h3, name='h3bn', is_training=train, decay=0.9)
            h3a = leaky_relu(h3bn)

            h4 = linear(h3a, 1024, 512, 'h4')
            h4bn = batch_norm_wrapper(h4, name='h4bn', is_training=train, decay=0.9)
            h4a = leaky_relu(h4bn)
            h5a = h4a

            if not self.use_tanh:
                out_mean = linear(h5a, 512, self.hidden_dim, name='out_mean')
                out_std = tf.sqrt(2 * tf.nn.sigmoid(linear(h5a, 512, self.hidden_dim, name='out_std')))
                return out_mean, out_std
            else:
                out_mean = linear(h5a, 512, self.hidden_dim, 'out_mean')# + latent_tensor
                return out_mean#tf.nn.tanh(out_mean)

    def dis_b(self, input_feature, train=True, reuse=False):
        with tf.variable_scope("dis/b") as scope:
            scope.set_regularizer(tf.contrib.layers.l2_regularizer(scale=1.0))
            if not train or reuse:
                train = False
                reuse = True
                scope.reuse_variables()

            conv1 = convlayer_pooling(input_feature, self.vertex_dim, self.vertex_dim, self.nb1_a, self.cw1_a,
                                      name='conv1', training=train)

            conv1 = convlayer_pooling(conv1, self.vertex_dim, self.finaldim, self.nb2_a, self.cw2_a,
                                      name='conv3', training=train, bn=False)
            l0 = tf.reshape(conv1, [tf.shape(conv1)[0], self.pointnum2_a * self.finaldim])
            if self.use_sigmoid == '2s':
                l0 = linear(l0, self.pointnum2_a * self.finaldim, 2048, 'dense1')
                return tf.nn.sigmoid(linear(l0, 2048, 1, 'dense2'))
            elif self.use_sigmoid == '1s':
                return tf.nn.sigmoid(linear(l0, self.pointnum2_a * self.finaldim, 1, 'dense1'))
            elif self.use_sigmoid == '2':
                l0 = linear(l0, self.pointnum2_a * self.finaldim, 2048, 'dense1')
                return linear(l0, 2048, 1, 'dense2')
            else:
                l0 = linear(l0, self.pointnum2_a * self.finaldim, 512, 'dense1')
                return l0

    def train_pre(self):
        tf.global_variables_initializer().run()
        if tb:
            self.write = tf.summary.FileWriter(self.logfolder + '/logs/', self.sess.graph)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        could_load, checkpoint_counter, Net_name = self.load(self.sess, self.checkpoint_dir)

        if could_load and Net_name == 'vae':
            self.start = 'VAE'
            self.start_step_vae = checkpoint_counter
            self.start_step_metric = 0
            self.start_step_gan = 0
        elif could_load and Net_name == 'metric':
            self.start = 'Mat'
            self.start_step_metric = checkpoint_counter
            self.start_step_gan = 0
        elif could_load and Net_name == 'gan':
            self.start = 'GAN'
            self.start_step_gan = checkpoint_counter
        else:
            print('we start from VAE...')

    def train_VAE(self):
        self.batch_size = 256
        rng = np.random.RandomState(23456)

        Ia = self.train_id_a
        Ib = self.train_id_b
        Ia_C = self.valid_id_a
        Ib_C = self.valid_id_b

        self.file.write("VAE start\n")
        for step in xrange(self.start_step_vae, self.n_epoch_Vae):
            timeserver1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            rng.shuffle(Ia)
            rng.shuffle(Ib)
            print("step%d: "%step, max(len(Ia), len(Ib)//self.batch_size+1))
            random_a_train = gaussian(len(Ia), self.hidden_dim)
            random_b_train = gaussian(len(Ib), self.hidden_dim)
            for bidx in xrange(0, max(len(Ia), len(Ib)) // self.batch_size + 1):

                feature_a = [self.feature_a[i] for i in Ia[(bidx * self.batch_size) % len(Ia):min(len(Ia), (
                            bidx * self.batch_size) % len(Ia) + self.batch_size)]]
                feature_b = [self.feature_b[i] for i in Ib[(bidx * self.batch_size) % len(Ib):min(len(Ib), (
                            bidx * self.batch_size) % len(Ib) + self.batch_size)]]
                random_a = [random_a_train[i] for i in Ia[(bidx * self.batch_size) % len(Ia):min(len(Ia), (
                            bidx * self.batch_size) % len(Ia) + self.batch_size)]]
                random_b = [random_b_train[i] for i in Ib[(bidx * self.batch_size) % len(Ib):min(len(Ib), (
                            bidx * self.batch_size) % len(Ib) + self.batch_size)]]
                if len(feature_a) == 0 or len(feature_b) == 0:
                    continue

                _, cost_generation_a, cost_latent_a, l2_loss_a, _, cost_generation_b, cost_latent_b, l2_loss_b = self.sess.run(
                    [self.train_op_vae_a, self.neg_loglikelihood_a, self.KL_divergence_a, self.r2_a,
                     self.train_op_vae_b, self.neg_loglikelihood_b, self.KL_divergence_b, self.r2_b],
                    feed_dict={self.inputs_a: feature_a, self.random_a: random_a, self.inputs_b: feature_b,
                               self.random_b: random_b})

                printout(self.file,
                         "|%s step: [%2d|%d]cost_generation_a: %.8f, cost_latent_a: %.8f, l2_loss_a: %.8f" % (
                             timeserver1, step + 1, self.n_epoch_Vae, cost_generation_a, cost_latent_a, l2_loss_a))

                printout(self.file,
                         "|%s step: [%2d|%d]cost_generation_b: %.8f, cost_latent_b: %.8f, l2_loss_b: %.8f" % (
                             timeserver1, step + 1, self.n_epoch_Vae, cost_generation_b, cost_latent_b, l2_loss_b))

            if tb and (step + 1) % 20 == 0:
                s = self.sess.run(self.merge_summary,
                                  feed_dict={self.inputs_a: feature_a, self.inputs_b: feature_b,
                                             self.random_a: random_a,
                                             self.random_b: random_b, self.lf_dis: Ilf})
                self.write.add_summary(s, step)

            if (step + 1) % 1000 == 0:
                print(self.logfolder)
                if test_vae:
                    self.test_vae(step)
                self.saver.save(self.sess, self.checkpoint_dir + '/vcgan_vae.model', global_step=step + 1)

        print('---------------------------------train VAE success!!----------------------------------')
        print('------------------------------------train Metric--------------------------------------')

    def train_metric_1(self):
        self.batch_size = 1024
        rng = np.random.RandomState(23456)

        Ia = np.array(self.train_id_a)
        Ib = np.array(self.train_id_b)
        Ia_C = self.valid_id_a
        Ib_C = self.valid_id_b

        self.file.write("Metric start step 1\n")
        for step in xrange(self.start_step_metric, self.n_epoch_Metric_1):
            timeserver1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            rng.shuffle(Ia)
            rng.shuffle(Ib)
            train_id_a = [random.choice(Ia) for _ in range(self.batch_size)]
            train_id_b = [random.choice(Ib) for _ in range(self.batch_size)]
            feature_a = self.feature_a[train_id_a]
            feature_b = self.feature_b[train_id_b]
            random_a = np.zeros((self.batch_size, self.hidden_dim)).astype('float32')
            random_b = np.zeros((self.batch_size, self.hidden_dim)).astype('float32')
            Ilf = np.expand_dims(self.lf_matrix[train_id_a, train_id_b], 1)

            _, cost_metric, cost_metric_l2, t_lf, t_dis = self.sess.run(
                [self.train_op_metric_1, self.loss_metric, self.loss_metric_l2, self.lf_dis, self.distance],
                feed_dict={self.inputs_a: feature_a, self.inputs_b: feature_b, self.random_a: random_a,
                           self.random_b: random_b, self.lf_dis: Ilf})

            printout(self.file, "|%s step: [%2d]cost_metric: %.8f,cost_metric_l2: %.8f" % (
                timeserver1, step + 1, cost_metric - cost_metric_l2, cost_metric_l2))
            print("gt:%.8f test:%.8f error:%.8f %.8f" % (
            t_lf[0], t_dis[0], np.max(abs(t_lf - t_dis)), np.mean(abs(t_lf - t_dis))))

            if tb and (step + 1) % 20 == 0:
                s = self.sess.run(self.merge_summary,
                                  feed_dict={self.inputs_a: feature_a, self.inputs_b: feature_b,
                                             self.random_a: random_a,
                                             self.random_b: random_b, self.lf_dis: Ilf})
                self.write.add_summary(s, step)

            if (step + 1) % 1000 == 0:
                print(self.logfolder)
                if test_metric:
                    self.test_metric(step + 1)
                self.saver.save(self.sess, self.checkpoint_dir + '/vcgan_metric.model', global_step=step + 1)

        print('---------------------------------train Metric success!!----------------------------------')
        print('------------------------------------train cycleGAN---------------------------------------')

    def train_metric(self):
        self.train_metric_1()

    def train_GAN(self):
        self.batch_size = 192
        rng = np.random.RandomState(23456)
        header = '     Time    Epoch      Progress(%)   cost_g   cost_mapping   cost_KL   cost_cycle   cost_d_all   cost_g_all'
        log_template = ' '.join(
            '{:>9s},{:>5.0f}/{:<5.0f},{:>9.1f}%,{:>11.2f},{:>10.2f},{:>10.2f},{:>11.2f},{:>12.2f},{:>15.2f}'.split(','))

        Gstep = 10
        Dstep = 0
        cost_d_all = 0.0
        cost_g = 100

        Ia = np.array(self.train_id_a)
        Ib = np.array(self.train_id_b)
        Ia_C = self.valid_id_a
        Ib_C = self.valid_id_b

        start = time.time()
        self.file.write("GAN start\n")
        for step in xrange(self.start_step_gan, self.n_epoch_Gan):
            if (step) % 100 == 0:
                print(header)
            timeserver1 = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

            rng.shuffle(Ia)
            rng.shuffle(Ib)
            train_id_a = [random.choice(Ia) for _ in range(self.batch_size)]
            train_id_b = [random.choice(Ib) for _ in range(self.batch_size)]
            feature_a = self.feature_a[train_id_a]
            feature_b = self.feature_b[train_id_b]
            random_a = gaussian(self.batch_size, self.hidden_dim)
            random_b = gaussian(self.batch_size, self.hidden_dim)
            Ilf = np.expand_dims(self.lf_matrix[train_id_a, train_id_b], 1)

            if cost_g > 00.0:
                for G in range(Gstep):
                    _, cost_mapping, cost_cycle, cost_g, cost_g_all, cost_d_all, cost_KL = self.sess.run(
                        [self.train_op_g, self.loss_mapping, self.loss_cycle, self.loss_g, self.loss_g_all,
                         self.loss_d_all, self.loss_KL],
                        feed_dict={self.inputs_a: self.feature_a, self.inputs_b: self.feature_b, self.random_a: random_a,
                                   self.random_b: random_b})
            if cost_d_all > 0.0:
                for D in range(Dstep):
                    _, cost_d_all, cost_mapping, cost_cycle, cost_g, cost_KL = self.sess.run(
                        [self.train_op_d, self.loss_d_all, self.loss_mapping, self.loss_cycle, self.loss_g,
                         self.loss_KL],
                        feed_dict={self.inputs_a: feature_a, self.inputs_b: feature_b, self.random_a: random_a,
                                   self.random_b: random_b})
            printout(self.file, log_template.format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start)),
                                                    step + 1, self.n_epoch_Gan, 100. * (step + 1) / self.n_epoch_Gan,
                                                    cost_g, cost_mapping, cost_KL, cost_cycle, cost_d_all, cost_g_all))

            if tb and (step + 1) % 2 == 0:
                s = self.sess.run(self.merge_summary,
                                  feed_dict={self.inputs_a: feature_a, self.inputs_b: feature_b,
                                             self.random_a: random_a,
                                             self.random_b: random_b, self.lf_dis: Ilf})
                self.write.add_summary(s, step)

            if (step + 0) % 100 == 0:
                print(self.logfolder)
                if test_gan:
                    self.test_gan(step + 1)
                self.saver.save(self.sess, self.checkpoint_dir + '/vcgan_gan.model', global_step=step + 0)

        print('---------------------------------train cycleGAN success!!----------------------------------\nFinished!')

    def train_all(self):
        with tf.Session(config=self.config) as self.sess:
            self.train_pre()

            if self.start == 'VAE':
                self.train_VAE()
                self.train_metric()
                self.train_GAN()
            elif self.start == 'Mat':
                self.train_metric()
                self.train_GAN()
            elif self.start == 'GAN':
                self.train_GAN()
            else:
                print('this is a train-finished model!!')
            print(self.logfolder)

    def test_vae(self, datainfo, step=0):
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            print(self.checkpoint_dir)
            _success, epoch, _ = self.load(sess, self.checkpoint_dir)
            if not _success:
                raise Exception("raise a problem")
            path = self.checkpoint_dir + '/../test_vae' + str(step)
            if not os.path.isdir(path):
                os.makedirs(path)

            zeros1a = np.zeros((self.model_num_a, self.hidden_dim)).astype('float32')
            zeros1b = np.zeros((self.model_num_b, self.hidden_dim)).astype('float32')
            z_meana, z_stddeva, recon_mesha, z_meanb, z_stddevb, recon_meshb = sess.run(
                [self.z_mean_a, self.z_stddev_a, self.generated_mesh_test_a,
                 self.z_mean_b, self.z_stddev_b, self.generated_mesh_test_b],
                feed_dict={self.inputs_a: self.feature_a, self.random_a: zeros1a, self.inputs_b: self.feature_b,
                           self.random_b: zeros1b})

            recon_fva = recover_data(recon_mesha, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a,
                                     self.logr_avg_a, self.s_avg_a, datainfo.resultmin,
                                     datainfo.resultmax, useS=datainfo.useS)
            recon_fvb = recover_data(recon_meshb, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b,
                                     self.logr_avg_b, self.s_avg_b, datainfo.resultmin,
                                     datainfo.resultmax, useS=datainfo.useS)

            v, _ = self.f2v_a.get_vertex(recon_fva, path + '/reconA')
            sio.savemat(path + '/reconA/recovera.mat',
                        {'feature': recon_fva, 'tid': self.train_id_a, 'vid': self.valid_id_a, 'latent_mean': z_meana,
                         'latent_std': z_stddeva})
            printout(self.file, "Erms: %.8f" % (self.f2v_a.calc_erms(v)))

            v, _ = self.f2v_b.get_vertex(recon_fvb, path + '/reconB')
            sio.savemat(path + '/reconB/recoverb.mat',
                        {'feature': recon_fvb, 'tid': self.train_id_b, 'vid': self.valid_id_b, 'latent_mean': z_meanb,
                         'latent_std': z_stddevb})
            printout(self.file, "Erms: %.8f" % (self.f2v_b.calc_erms(v)))

            random_batch_a = np.random.normal(loc=0.0, scale=1.0, size=(50, self.hidden_dim))
            random_batch_b = np.random.normal(loc=0.0, scale=1.0, size=(50, self.hidden_dim))
            testa, testb = sess.run([self.test_mesh_a, self.test_mesh_b],
                                    feed_dict={self.random_a: random_batch_a, self.random_b: random_batch_b})
            fv1a = recover_data(testa, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                                self.s_avg_a, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            fv1b = recover_data(testb, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                                self.s_avg_b, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)

            self.f2v_a.get_vertex(fv1a, path + '/randomA', one=True)
            sio.savemat(path + '/randomA/randoma.mat', {'feature': fv1a, 'tid': self.train_id_a, 'vid': self.valid_id_a,
                                                        'latent_mean': random_batch_a})

            self.f2v_b.get_vertex(fv1b, path + '/randomB', one=True)
            sio.savemat(path + '/randomB/randomb.mat', {'feature': fv1b, 'tid': self.train_id_b, 'vid': self.valid_id_b,
                                                        'latent_mean': random_batch_b})

    def test_gan(self, datainfo, step=0):
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            _success, epoch, _ = self.load(sess, self.checkpoint_dir)
            if not _success:
                raise Exception("raise a problem")
            path = self.checkpoint_dir + '/../test_gan' + str(step)
            if not os.path.isdir(path):
                os.makedirs(path)

            random_a = gaussian(self.batch_size, self.hidden_dim)
            random_b = gaussian(self.batch_size, self.hidden_dim)
            a_gen_b, b_gen_a = sess.run([self.gen_feature_a, self.gen_feature_b],
                                        feed_dict={self.inputs_a: self.feature_a, self.inputs_b: self.feature_b,
                                                   self.random_a: random_a, self.random_b: random_b})
            IA = recover_data(self.feature_a, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                              self.s_avg_a, datainfo.resultmin,
                              datainfo.resultmax, useS=datainfo.useS)
            IB = recover_data(self.feature_b, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                              self.s_avg_b, datainfo.resultmin,
                              datainfo.resultmax, useS=datainfo.useS)
            a_gen_b1 = recover_data(a_gen_b, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                                    self.s_avg_b, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            b_gen_a1 = recover_data(b_gen_a, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                                    self.s_avg_a, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)

            self.f2v_a.get_vertex(IA, path + '/AtoB/A', one=True)
            sio.savemat(path + '/AtoB/A.mat', {'feature': IA})

            self.f2v_b.get_vertex(a_gen_b1, path + '/AtoB/B', one=True)
            sio.savemat(path + '/AtoB/B.mat', {'feature': a_gen_b1})

            self.f2v_b.get_vertex(IB, path + '/BtoA/B', one=True)
            sio.savemat(path + '/BtoA/B.mat', {'feature': IB})

            self.f2v_a.get_vertex(b_gen_a1, path + '/BtoA/A', one=True)
            sio.savemat(path + '/BtoA/A.mat', {'feature': b_gen_a1})

    def test_metric(self, datainfo, step=0):
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            _success, epoch, _ = self.load(sess, self.checkpoint_dir)
            if not _success:
                raise Exception("raise a problem")
            path = self.checkpoint_dir + '/../test_metric' + str(step)
            if not os.path.isdir(path):
                os.makedirs(path)

            random_batch_a = np.random.normal(loc=0.0, scale=1.0, size=(50, self.hidden_dim))
            random_batch_b = np.random.normal(loc=0.0, scale=1.0, size=(50, self.hidden_dim))
            testa, testb, test_dis = sess.run([self.test_mesh_a, self.test_mesh_b, self.distance_test],
                                              feed_dict={self.random_a: random_batch_a, self.random_b: random_batch_b})
            fv1b = recover_data(testb, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                                self.s_avg_b, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            fv1a = recover_data(testa, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                                self.s_avg_a, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            test_dis = recover_data_old(test_dis, self.lf_matrix_min, self.lf_matrix_max, 0.05, 1.95)

            self.f2v_a.get_vertex(fv1a, path + '/A', one=True)
            sio.savemat(path + '/A/A.mat', {'feature': fv1a})

            self.f2v_b.get_vertex(fv1b, path + '/B', one=True)
            sio.savemat(path + '/B/B.mat', {'feature': fv1b})
            sio.savemat(path + '/pre_dis.mat', {'test_dis': test_dis})

            za, zb = sess.run([self.z_mean_a, self.z_mean_b],
                              feed_dict={self.inputs_a: self.feature_a, self.inputs_b: self.feature_b})
            id = np.min([np.shape(za)[0], np.shape(zb)[0]])
            testa, testb, test_dis = sess.run([self.test_mesh_a, self.test_mesh_b, self.distance_test],
                                              feed_dict={self.random_a: za[0:id], self.random_b: zb[0:id]})
            fv1b = recover_data(testb, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                                self.s_avg_b, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            fv1a = recover_data(testa, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                                self.s_avg_a, datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            test_dis = recover_data_old(test_dis, self.lf_matrix_min, self.lf_matrix_max, 0.05, 1.95)

            self.f2v_a.get_vertex(fv1a, path + '/oriA')
            sio.savemat(path + '/oriA/A.mat', {'feature': fv1a})

            self.f2v_b.get_vertex(fv1b, path + '/oriB')
            sio.savemat(path + '/oriB/B.mat', {'feature': fv1b})
            sio.savemat(path + '/pre_dis_ori.mat', {'test_dis': test_dis})

    def test_cycle(self, datainfo, step=0):
        with tf.Session(config=self.config) as sess:
            tf.global_variables_initializer().run()
            _success, epoch, _ = self.load(sess, self.checkpoint_dir)
            if not _success:
                raise Exception("raise a problem")
            path = self.checkpoint_dir + '/../test_cycle' + str(step)
            if not os.path.isdir(path):
                os.makedirs(path)

            random_a = gaussian(self.batch_size, self.hidden_dim)
            random_b = gaussian(self.batch_size, self.hidden_dim)

            latent_zx, decoder_x, g_latent_zx, decoder_gx, fg_latent_zx, decoder_fgx = sess.run(
                [self.z_mean_test_a, self.decoder_x, self.gen_latent_a, self.decoder_gx, self.cycle_z_a,
                 self.decoder_fgx],
                feed_dict={self.inputs_a: self.feature_a, self.inputs_b: self.feature_b,
                           self.random_a: random_a, self.random_b: random_b})
            latent_zy, decoder_y, g_latent_zy, decoder_gy, fg_latent_zy, decoder_fgy = sess.run(
                [self.z_mean_test_b, self.decoder_y, self.gen_latent_b, self.decoder_fy, self.cycle_z_b,
                 self.decoder_gfy],
                feed_dict={self.inputs_a: self.feature_a, self.inputs_b: self.feature_b,
                           self.random_a: random_a, self.random_b: random_b})
            IA = recover_data(self.feature_a, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a, self.logr_avg_a,
                              self.s_avg_a,
                              datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            IB = recover_data(self.feature_b, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b, self.logr_avg_b,
                              self.s_avg_b,
                              datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)

            decoder_y1 = recover_data(decoder_y, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b,
                                      self.logr_avg_b, self.s_avg_b,
                                      datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            decoder_x1 = recover_data(decoder_x, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a,
                                      self.logr_avg_a, self.s_avg_a,
                                      datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            decoder_gx1 = recover_data(decoder_gx, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b,
                                       self.logr_avg_b, self.s_avg_b,
                                       datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            decoder_gy1 = recover_data(decoder_gy, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a,
                                       self.logr_avg_a, self.s_avg_a,
                                       datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            decoder_fgy1 = recover_data(decoder_fgy, self.logrmin_b, self.logrmax_b, self.smin_b, self.smax_b,
                                        self.logr_avg_b, self.s_avg_b,
                                        datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)
            decoder_fgx1 = recover_data(decoder_fgx, self.logrmin_a, self.logrmax_a, self.smin_a, self.smax_a,
                                        self.logr_avg_a, self.s_avg_a,
                                        datainfo.resultmin, datainfo.resultmax, useS=datainfo.useS)

            self.f2v_a.get_vertex(IA, path + '/groundtruthA')
            sio.savemat(path + '/groundtruthA.mat', {'feature': IA})

            self.f2v_b.get_vertex(IB, path + '/groundtruthB')
            sio.savemat(path + '/groundtruthB.mat', {'feature': IB})

            self.f2v_a.get_vertex(decoder_x1, path + '/decoder_xA')
            sio.savemat(path + '/decoder_xA.mat', {'feature': decoder_x1, 'latent_z': latent_zx})

            self.f2v_b.get_vertex(decoder_gx1, path + '/decoder_gxA')
            sio.savemat(path + '/decoder_gxA.mat', {'feature': decoder_gx1, 'latent_z': g_latent_zx})

            self.f2v_a.get_vertex(decoder_fgx1, path + '/decoder_fgxA')
            sio.savemat(path + '/decoder_fgxA.mat', {'feature': decoder_fgx1, 'latent_z': fg_latent_zx})

            self.f2v_b.get_vertex(decoder_y1, path + '/decoder_yB')
            sio.savemat(path + '/decoder_yB.mat', {'feature': decoder_y1, 'latent_z': latent_zy})

            self.f2v_a.get_vertex(decoder_gy1, path + '/decoder_gyB')
            sio.savemat(path + '/decoder_gyB.mat', {'feature': decoder_gy1, 'latent_z': g_latent_zy})

            self.f2v_b.get_vertex(decoder_fgy1, path + '/decoder_fgyB')
            sio.savemat(path + '/decoder_fgyB.mat', {'feature': decoder_fgy1, 'latent_z': fg_latent_zy})


    def model_dir(self, model_name, dataset_name):
        return "{}_{}_{}_{}".format(model_name, dataset_name,
                                    self.batch_size, self.hidden_dim)

    def load(self, sess, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        if not checkpoint_dir.find(self.VAE) == -1:
            saver = self.saver
        elif not checkpoint_dir.find(self.METRIC) == -1:
            saver = self.saver
        else:
            saver = self.saver

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        from tensorflow.python.tools import inspect_checkpoint as chkp
        ckpt_name = ''
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter, ckpt_name.split('.')[0].split('_')[1]
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0, ckpt_name
