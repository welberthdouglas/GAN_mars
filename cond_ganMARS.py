#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 15:55:31 2020

@author: welberth
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 22:02:13 2020

@author: welberth
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd


from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot
from keras.models import load_model


import keras.backend as K
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import Sequential, Input, Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import TensorBoard
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
#from scipy.misc import imread, imsave
from scipy.stats import entropy

from keras.utils.vis_utils import plot_model


#d_learning_rate = 0.005
#g_learning_rate = 0.005
#d_momentum = 0.5
#g_momentum = 0.5
#d_nesterov = True
#g_nesterov = True
#
#d_optimizer = SGD(lr=d_learning_rate, momentum=d_momentum, nesterov=d_nesterov)
#g_optimizer = SGD(lr=g_learning_rate, momentum=g_momentum, nesterov=g_nesterov)





# Discriminator

def define_discriminator(input_shape = (128,128,1),n_classes=8):
    
    # label input
    in_label = Input(shape=(1,))
    
    #embeding for categorical input
    input_l = Embedding(n_classes,50)(in_label)
    
    #scale_up to image shape and reshape to image size
    input_l = Dense(input_shape[0]*input_shape[1])(input_l)
    input_l = Reshape((input_shape[0],input_shape[1],1))(input_l)
    
    #image input
    input_img = Input(shape=input_shape)
    
    #adding the label input as a channel to the image
    img_plus_label = Concatenate()([input_img,input_l])
    
    
    #normal layers
    l1 = Conv2D(128, (5,5), padding='same', input_shape=input_shape)(img_plus_label)
    l1 = LeakyReLU(alpha=0.2)(l1)
    l1 = MaxPooling2D(pool_size=(2, 2))(l1)
    
    l2 = Conv2D(256, (3, 3))(l1)
    l2 = LeakyReLU(alpha=0.2)(l2)
    l2 = MaxPooling2D(pool_size=(2, 2))(l2)
    
    l3 = Conv2D(512, (3, 3))(l2)
    l3 = LeakyReLU(alpha=0.2)(l3)
    l3 = MaxPooling2D(pool_size=(2, 2))(l3)
    
    l4 = Flatten()(l3)
    l4 = Dense(1024)(l4)
    l4 = LeakyReLU(alpha=0.2)(l4)
    
    l5 = Dense(1, activation='sigmoid')(l4)
    
    #define model
    d_model = Model([input_img,in_label],l5)
    
    #compile
    opt = Adam(lr=0.0002, beta_1=0.5)
    d_model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])

    return d_model

def define_generator(input_dim,n_classes=8):

    # label input
    in_label = Input(shape=(1,))
    
    # embedding
    input_l = Embedding(n_classes,50)(in_label)
    
    # upsampling and reshape
    input_l = Dense(8*8)(input_l)
    input_l = Reshape((8,8,1))(input_l)
    
    
    # image generator input
    in_img_gen = Input(shape=(input_dim,))
    
    input_gen = Dense(2048,input_dim=input_dim)(in_img_gen)
    input_gen = ReLU()(input_gen)
    

    input_gen = Dense(256 * 8 * 8)(input_gen)
    input_gen = BatchNormalization()(input_gen)
    input_gen = ReLU()(input_gen)
    input_gen = Reshape((8,8,256))(input_gen)
    
    # adding label as feature chanel
    
    gen_plus_label = Concatenate()([input_l,input_gen])
    
    # upsample to 16X16
    l1 = Conv2D(256, (5,5), padding='same')(gen_plus_label)
    l1 = ReLU()(l1)
    l1 = UpSampling2D(size=(2,2))(l1)


    # upsample to 32X32
    l2 = Conv2D(128, (5,5), padding='same')(l1)
    l2 = ReLU()(l2)
    l2 = UpSampling2D(size=(2,2))(l2)
    
    # upsample to 64X64
    l3 = Conv2D(64, (5,5), padding='same')(l2)
    l3 = ReLU()(l3)
    l3 = UpSampling2D(size=(2,2))(l3)

    # upsample to 128X128
    l4 = Conv2D(32, (5,5), padding='same')(l3)
    l4 = ReLU()(l4)
    l4 = UpSampling2D(size=(2,2))(l4)

    #output layer
    l5 = Conv2D(1, (5,5), activation='tanh', padding='same')(l4)
    
    # define model
    g_model = Model([in_label,in_img_gen],l5)
    
    
    return g_model

    
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False

    # latent space vector and label as inputs
    g_label,g_lat_noise = g_model.input
    
    # output of the generator
    g_img_output = g_model.output
    
    # connect image output and label input from the generator as inputs to the discriminator
    gan_output = d_model([g_img_output,g_label])
    
    # define the model as taking latent space vector and label as input and outputting a classification
    model = Model([g_lat_noise,g_label],gan_output)
    
    #g_model.compile(loss='mse', optimizer=g_optimizer)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space

    x_input = np.random.normal(0, 1, size=(n_samples, latent_dim))
    y_input = np.random.randint(0,8,size = n_samples)
    return [x_input,y_input]


def generate_fake_samples(g_model, latent_dim, batch_size, binary=False):
    # generate points in latent space
    x_input,y_input = generate_latent_points(latent_dim, batch_size)
    # predict outputs
    X = g_model.predict([y_input,x_input])
    # create 'fake' class labels (0)
    #y = zeros((batch_size, 1))
    
    if binary:
       y = zeros((batch_size, 1))
    else:
       y = zeros((batch_size, 1))-np.random.random_sample((batch_size,1)) * 0.2
    
    return X, y_input,y




#def load_images():
#    
#    path="croped_pics/"
#    X = (expand_dims(np.array([cv2.imread(p,0) for p in glob.glob(path+'*.jpg')]),axis=-1)-127.5)/127.5
#    X = X.astype('float32')
#    X = X[0:10000]
#    return X



def load_images():
    
    path="croped_pics/"
    img_list = glob.glob(path+'*')
    
    img_list = [i for i in img_list if 'brt.jpg' not in i]  # excludes images with random brightness
    
    #labels_dict = pd.read_csv('landmarks_map-proj-v3_classmap.csv', names=['class_number','class_name'],index_col='class_number')
    labels = pd.read_csv('labels-map-proj-v3.txt',sep=' ',names=['file','class'])
    img_df = pd.DataFrame([i[12:] for i in img_list],columns=['file'])
    img_df=img_df.merge(labels, on='file', how='left')
    
    X = expand_dims(np.array([cv2.normalize(cv2.imread(path+i,0), None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for i in img_df.file]),axis = -1)
    X = X.astype('float32')
    
    Y = np.array(img_df['class'])
    
    #Y = Y[0:200]
    #X = X[0:200]
    return [X,Y]

# select real samples
def generate_real_samples(dataset,labels,index, batch_size, binary=False):

    X = dataset[index * batch_size:(index + 1) * batch_size]
    label = labels[index * batch_size:(index + 1) * batch_size]
    # generate 'real' class labels (1)
    #y_2 = ones((batch_size, 1))
    if binary:
       y = ones((batch_size,1))
    else:
       y = ones((batch_size,1))- np.random.random_sample((batch_size,1)) * 0.2
    return X,label,y


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch)
    
    
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real,label_real,y_real = generate_real_samples(dataset,labels,j, half_batch)
            # generate 'fake' examples
            X_fake,label_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            
            # train discriminator
            dis_loss_real,acc_r = d_model.train_on_batch([X_real,label_real], y_real)
            dis_loss_fake,acc_f = d_model.train_on_batch([X_fake,label_fake], y_fake)
            
            d_loss = (dis_loss_real+dis_loss_fake)/2

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            #y_gan = ones((n_batch, 1))
            y_gan = ones((n_batch,1))#*0.9# - np.random.random_sample((n_batch,1)) * 0.2
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d_loss=%.3f, gan_loss=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
            
            x_fake_pic,label_fake_pic,_ = generate_fake_samples(g_model, latent_dim, 4)
            
            for i in range(4):
                # define subplot
                pyplot.subplot(2, 2, 1 + i)
                # turn off axis
                pyplot.axis('off')
                pyplot.title(f'{label_fake_pic[i]}')
                # plot raw pixel data
                pyplot.imshow(x_fake_pic[i, :, :, 0], cmap='gray_r')
            # save plot to file
            filename = 'generated_plot_batch{}.png'.format(j)
            pyplot.savefig(filename)
        # evaluate the model performance, sometimes
        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, d_model, dataset,labels, latent_dim)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples,labels,epoch, n=4):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')

        pyplot.title(f'{labels[i]}')
        pyplot.tight_layout()
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


def random_real_samples(dataset,labels, n_samples, binary = False):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    label = labels[ix]
    # generate 'real' class labels (1)
    if binary:
       y = ones((n_samples,1))
    else:
       y = ones((n_samples,1))*0.9#- np.random.random_sample((n_samples,1)) * 0.2
    return X,label,y


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset,labels,latent_dim, n_samples=100,n_plots=16,save=True):
    # prepare real samples
    X_real,label_real,y_real = random_real_samples(dataset,labels, n_samples)
    # evaluate discriminator on real examples
    _,acc_real = d_model.evaluate([X_real,label_real], np.around(y_real), verbose=0)
    # prepare fake examples
    x_fake,label_fake,y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _,acc_fake = d_model.evaluate([x_fake,label_fake], np.around(y_fake), verbose=0)
    # summarize discriminator performance
    print('>Discriminator accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake[0:n_plots],label_fake[0:n_plots], epoch)
    # save the generator and discriminator model file
    if save:
        g_filename = 'generator_model_%03d.h5' % (epoch + 1)
        d_filename = 'discriminator_model_%03d.h5' % (epoch + 1)
        g_model.save(g_filename)
        d_model.save(d_filename)


# size of the latent space
latent_dim = 300
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)

# Load models

#d_model = load_model('discriminator_model_004.h5')
#g_model = load_model('generator_model_004.h5')

# create the gan
gan_model = define_gan(g_model, d_model)    
# load image data
dataset,labels = load_images()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

def show_img(img):
    pyplot.imshow(img[0, :, :, 0], cmap='gray_r')

#plot_model(g_model, to_file='conditional_generator_plot.png', show_shapes=True, show_layer_names=True)
#plot_model(d_model, to_file='conditional_discriminator_plot.png', show_shapes=True, show_layer_names=True)
#plot_model(gan_model, to_file='conditional_gan_plot.png', show_shapes=True, show_layer_names=True)
