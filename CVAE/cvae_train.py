#! /usr/bin/python
# -*- coding: utf8 -*-

""" GAN-CLS """
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk
import math
from utils import *
from model import *
import model


###======================== PREPARE DATA ====================================###
print("Loading data from pickle ...")
import pickle
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)

images_train = np.array(images_train)
images_test = np.array(images_test)

ni = int(np.ceil(np.sqrt(batch_size)))
tl.files.exists_or_mkdir("samples/step1_gan-cls")
tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
tl.files.exists_or_mkdir("checkpoint")
save_dir = "checkpoint"


def main_train():
    t_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
    t_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')

    # Loading the text-encoder
    net_rnn = rnn_embed(t_caption, is_train=False, reuse=False)
    text_encoded = net_rnn.outputs

    # Creating the VAE Model
    encoder = model.vae_encoder
    decoder = model.vae_decoder
    sampling = model.sampling

    net_encoder, sampled, mn, sd = encoder(t_image, text_encoded)
    encoded = sampled
    net_decoder = decoder(sampled, text_encoded)
    output_images = net_decoder

    n_latent = 28
    #x = DenseLayer(encoded, n_units=16, act=tf.nn.relu)
    #z_mean = DenseLayer(x, n_units=latent_dim, name='z_mean')
    #z_log_var = DenseLayer(x, n_units=latent_dim, name='z_log_var')
    #z = sampling(z_mean.outputs, z_log_var.outputs)

    '''reconstruction_loss = tl.cost.sigmoid_cross_entropy(t_image, output_images.outputs, name='loss1')
    kl_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_log_var - tf.square(z_mean) - tf.exp(2.0 * z_log_var), 1)
    vae_loss = (reconstruction_loss + kl_loss)/2.0
    '''
    #unreshaped = tf.reshape(dec, [-1, 28*28])
    img_loss = tf.reduce_sum(tf.squared_difference(t_image, output_images.outputs), 1)
    img_loss = tf.math.reduce_sum(img_loss, axis=1)
    img_loss = tf.math.reduce_sum(img_loss, axis=1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
    #print (img_loss)
    #print (latent_loss)
    vae_loss = tf.reduce_mean(img_loss + latent_loss)

    lr = 0.002
    lr_decay = 0.5      # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100   # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
    vae_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(vae_loss)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    net_encoder_name = os.path.join(save_dir, 'net_encoder.npz')
    net_decoder_name = os.path.join(save_dir, 'net_decoder.npz')
    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')

    load_and_assign_npz(sess=sess, name=net_encoder_name, model=net_encoder)
    load_and_assign_npz(sess=sess, name=net_decoder_name, model=net_decoder)
    load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)


    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
    n = int(sample_size / ni)
    sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * n + \
                      ["this flower has petals that are yellow, white and purple and has dark lines"] * n + \
                      ["the petals on this flower are white with a yellow center"] * n + \
                      ["this flower has a lot of small round pink petals."] * n + \
                      ["this flower is orange in color, and has petals that are ruffled and rounded."] * n + \
                      ["the flower has yellow petals and the center of it is brown."] * n + \
                      ["this flower has petals that are blue and white."] * n +\
                      ["these white flowers have petals that start off white in color and end in a white towards the tips."] * n

    for i, sentence in enumerate(sample_sentence):
        print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]    # add END_ID

    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    n_epoch = 150
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)
    
    for epoch in range(0, n_epoch+1):
        start_time = time.time()

        '''if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
            # logging.debug(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
        '''
        for step in range(n_batch_epoch):
            step_time = time.time()

            idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
            b_caption = captions_ids_train[idexs]
            b_caption = tl.prepro.pad_sequences(b_caption, padding='post')
            b_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
            b_images = threading_data(b_images, prepro_img, mode='train')   
            z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

            ## updates D
            err, _ , re_loss, kl = sess.run([vae_loss, vae_optim, img_loss, latent_loss], feed_dict={
                            t_image : b_images,
                            t_caption : b_caption})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, loss: %.8f" \
                        % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, err))
            

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time()-start_time))

            sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, n_latent)).astype(np.float32)
            imgs = sess.run(output_images.outputs, feed_dict={
                                        t_caption : sample_sentence,
                                        sampled : sample_seed})
            imgs = np.array(imgs)
            #print (imgs.shape)
            save_images(imgs, [ni, ni], 'samples/step1_gan-cls/train_{:02d}.png'.format(epoch))

        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(net_encoder.all_params, name=net_encoder_name, sess=sess)
            tl.files.save_npz(net_decoder.all_params, name=net_decoder_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train",
                       help='train, train_encoder, translation')
    args = parser.parse_args()
    if args.mode == "train":
        main_train()
