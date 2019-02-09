#! /usr/bin/python
# -*- coding: utf8 -*-



import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import math
import numpy as np

"""Adversarially Learned Inference
Page 14: CelebA model hyperparameters
Optimizer Adam (α = 10−4, β1 = 0.5)
Batch size 100 Epochs 123
Leaky ReLU slope 0.02
Weight, bias initialization Isotropic gaussian (µ = 0, σ = 0.01), Constant(0)
"""
batch_size = 64

z_dim = 512         # Noise dimension
image_size = 64     # 64 x 64
c_dim = 3           # for rgb

## for text-to-image mapping ===================================================
t_dim = 128         # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0

def rnn_embed(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')
        return network

def cnn_encoder(inputs, is_train=True, reuse=False, name='cnnftxt', return_h3=False):
    """ 64x64 --> t_dim, for text-image mapping """
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64

    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(True)

        net_in = InputLayer(inputs, name='/in')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='cnnf/h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h1/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h1 = DropoutLayer(net_h1, keep=0.8, is_fix=True, name='p/h1/drop')

        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h2/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h2 = DropoutLayer(net_h2, keep=0.8, is_fix=True, name='p/h2/drop')

        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='cnnf/h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='cnnf/h3/batch_norm')

        # if name != 'cnn': # debug for training image encoder in step 2
        #     net_h3 = DropoutLayer(net_h3, keep=0.8, is_fix=True, name='p/h3/drop')

        net_h4 = FlattenLayer(net_h3, name='cnnf/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units= (z_dim if name == 'z_encoder' else t_dim),
                act=tf.identity,
                W_init = w_init, b_init = None, name='cnnf/h4/embed')
    if return_h3:
        return net_h4, net_h3
    else:
        return net_h4


def vae_decoder(input_encoded, t_txt=None, is_train=True, reuse=False, batch_size=batch_size):
    """ z + (txt) --> 64x64 """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    s = image_size # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    gf_dim = 128

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_encoded, name='g_inputz')

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='g_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init, name='g_reduce_text/dense')
            net_in = ConcatLayer([net_in, net_txt], concat_dim=1, name='g_concat_z_txt')

        net_h0 = DenseLayer(net_in, gf_dim*8*s16*s16, act=tf.identity,
                W_init=w_init, b_init=None, name='g_h0/dense')
        net_h0 = BatchNormLayer(net_h0,  #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h0/batch_norm')
        net_h0 = ReshapeLayer(net_h0, [-1, s16, s16, gf_dim*8], name='g_h0/reshape')

        net = Conv2d(net_h0, gf_dim*2, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm')
        net = Conv2d(net, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h1_res/batch_norm2')
        net = Conv2d(net, gf_dim*8, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h1_res/conv2d3')
        net = BatchNormLayer(net, # act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h1_res/batch_norm3')
        net_h1 = ElementwiseLayer(layer=[net_h0, net], combine_fn=tf.add, name='g_h1_res/add')
        net_h1.outputs = tf.nn.relu(net_h1.outputs)

        # Note: you can also use DeConv2d to replace UpSampling2dLayer and Conv2d
        # net_h2 = DeConv2d(net_h1, gf_dim*4, (4, 4), out_size=(s8, s8), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h2/decon2d')
        net_h2 = UpSampling2dLayer(net_h1, size=[s8, s8], is_scale=False, method=1,
                align_corners=False, name='g_h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2,# act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h2/batch_norm')

        net = Conv2d(net_h2, gf_dim, (1, 1), (1, 1),
                padding='VALID', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm')
        net = Conv2d(net, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d2')
        net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,
                gamma_init=gamma_init, name='g_h3_res/batch_norm2')
        net = Conv2d(net, gf_dim*4, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h3_res/conv2d3')
        net = BatchNormLayer(net, #act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h3_res/batch_norm3')
        net_h3 = ElementwiseLayer(layer=[net_h2, net], combine_fn=tf.add, name='g_h3/add')
        net_h3.outputs = tf.nn.relu(net_h3.outputs)

        # net_h4 = DeConv2d(net_h3, gf_dim*2, (4, 4), out_size=(s4, s4), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h4/decon2d'),
        net_h4 = UpSampling2dLayer(net_h3, size=[s4, s4], is_scale=False, method=1,
                align_corners=False, name='g_h4/upsample2d')
        net_h4 = Conv2d(net_h4, gf_dim*2, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h4/conv2d')
        net_h4 = BatchNormLayer(net_h4, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h4/batch_norm')

        # net_h5 = DeConv2d(net_h4, gf_dim, (4, 4), out_size=(s2, s2), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, b_init=b_init, name='g_h5/decon2d')
        net_h5 = UpSampling2dLayer(net_h4, size=[s2, s2], is_scale=False, method=1,
                align_corners=False, name='g_h5/upsample2d')
        net_h5 = Conv2d(net_h5, gf_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, b_init=None, name='g_h5/conv2d')
        net_h5 = BatchNormLayer(net_h5, act=tf.nn.relu,
                is_train=is_train, gamma_init=gamma_init, name='g_h5/batch_norm')

        # net_ho = DeConv2d(net_h5, c_dim, (4, 4), out_size=(s, s), strides=(2, 2),
        #         padding='SAME', batch_size=batch_size, act=None, W_init=w_init, name='g_ho/decon2d')
        net_ho = UpSampling2dLayer(net_h5, size=[s, s], is_scale=False, method=1,
                align_corners=False, name='g_ho/upsample2d')
        net_ho = Conv2d(net_ho, c_dim, (3, 3), (1, 1),
                padding='SAME', act=None, W_init=w_init, name='g_ho/conv2d')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.tanh(net_ho.outputs)
    return net_ho

def vae_encoder(input_images, t_txt=None, is_train=True, reuse=False):
    """ 64x64 + (txt) --> real/fake """
    # https://github.com/hanzhanggit/StackGAN/blob/master/stageI/model.py
    # Discriminator with ResNet : line 197 https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.02)
    df_dim = 64  # 64 for flower, 196 for MSCOCO
    s = 64 # output image size [64]
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
    n_latent = 28

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='d_input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lambda x: tl.act.lrelu(x, 0.2),
                padding='SAME', W_init=w_init, name='d_h0/conv2d')

        net_h1 = Conv2d(net_h0, df_dim*2, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h1/batchnorm')
        net_h2 = Conv2d(net_h1, df_dim*4, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h2/batchnorm')
        net_h3 = Conv2d(net_h2, df_dim*8, (4, 4), (2, 2), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h3/conv2d')
        net_h3 = BatchNormLayer(net_h3, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h3/batchnorm')

        net = Conv2d(net_h3, df_dim*2, (1, 1), (1, 1), act=None,
                padding='VALID', W_init=w_init, b_init=None, name='d_h4_res/conv2d')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm')
        net = Conv2d(net, df_dim*2, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d2')
        net = BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm2')
        net = Conv2d(net, df_dim*8, (3, 3), (1, 1), act=None,
                padding='SAME', W_init=w_init, b_init=None, name='d_h4_res/conv2d3')
        net = BatchNormLayer(net, #act=lambda x: tl.act.lrelu(x, 0.2),
                is_train=is_train, gamma_init=gamma_init, name='d_h4_res/batchnorm3')
        net_h4 = ElementwiseLayer(layer=[net_h3, net], combine_fn=tf.add, name='d_h4/add')
        net_h4.outputs = tl.act.lrelu(net_h4.outputs, 0.2)

        if t_txt is not None:
            net_txt = InputLayer(t_txt, name='d_input_txt')
            net_txt = DenseLayer(net_txt, n_units=t_dim,
                   act=lambda x: tl.act.lrelu(x, 0.2),
                   W_init=w_init, name='d_reduce_txt/dense')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')
            net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
            net_txt = TileLayer(net_txt, [1, 4, 4, 1], name='d_txt/tile')
            net_h4_concat = ConcatLayer([net_h4, net_txt], concat_dim=3, name='d_h3_concat')
            # 243 (ndf*8 + 128 or 256) x 4 x 4
            net_h4 = Conv2d(net_h4_concat, df_dim*8, (1, 1), (1, 1),
                    padding='VALID', W_init=w_init, b_init=None, name='d_h3/conv2d_2')
            net_h4 = BatchNormLayer(net_h4, act=lambda x: tl.act.lrelu(x, 0.2),
                    is_train=is_train, gamma_init=gamma_init, name='d_h3/batch_norm_2')

        net_ho = Conv2d(net_h4, 1, (s16, s16), (s16, s16), padding='VALID', W_init=w_init, name='d_ho/conv2d')
        net_ho = tl.layers.FlattenLayer(net_ho, name='flatten')
        #net_ho = tf.layers.dense(net_ho, units=n_latent)

        '''mn = tf.layers.dense(net_ho, units=n_latent)
        sd       = 0.5 * tf.layers.dense(net_ho, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(net_ho)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))
        '''
        x = tf.contrib.layers.flatten(net_ho.outputs)
        mn = tf.layers.dense(x, units=n_latent)
        sd = 0.5 * tf.layers.dense(x, units=n_latent)            
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent])) 
        z  = mn + tf.multiply(epsilon, tf.exp(sd))

    return net_ho, z, mn, sd

def sampling(z_mean, z_log_var):
    batch = np.shape(z_mean)[0]
    dim = np.shape(z_mean)[1]
    epsilon = np.random.normal(0, 0.1, (batch, dim))
    print (type(z_mean))
    print (type(z_log_var))
    return z_mean + math.exp(0.5 * z_log_var) * epsilon