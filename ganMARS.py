import os
import glob
import cv2
import numpy as np


from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

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


d_learning_rate = 0.005
g_learning_rate = 0.005
d_momentum = 0.5
g_momentum = 0.5
d_nesterov = True
g_nesterov = True

d_optimizer = SGD(lr=d_learning_rate, momentum=d_momentum, nesterov=d_nesterov)
g_optimizer = SGD(lr=g_learning_rate, momentum=g_momentum, nesterov=g_nesterov)





# Discriminator

def define_discriminator(input_shape = (128,128,1)):

    d_model = Sequential()
    
    d_model.add(Conv2D(128, (5,5), padding='same', input_shape=input_shape))
    d_model.add(LeakyReLU(alpha=0.2))
    d_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    d_model.add(Conv2D(256, (3, 3)))
    d_model.add(LeakyReLU(alpha=0.2))
    d_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    d_model.add(Conv2D(512, (3, 3)))
    d_model.add(LeakyReLU(alpha=0.2))
    d_model.add(MaxPooling2D(pool_size=(2, 2)))
    
    d_model.add(Flatten())
    d_model.add(Dense(1024))
    d_model.add(LeakyReLU(alpha=0.2))
    
    d_model.add(Dense(1, activation='sigmoid'))
    
    d_model.compile(loss='binary_crossentropy', optimizer=d_optimizer)

    return d_model

def define_generator(input_dim):

    g_model = Sequential()
    g_model.add(Dense(2048,input_dim=input_dim))
    g_model.add(LeakyReLU(alpha=0.2))

    g_model.add(Dense(256 * 8 * 8))
    g_model.add(BatchNormalization())
    g_model.add(LeakyReLU(alpha=0.2))
    
    # upsample to 16X16
    g_model.add(Reshape((8, 8, 256)))
    g_model.add(Conv2DTranspose(256, (4,4), strides=(2,2), padding='same'))
    g_model.add(LeakyReLU(alpha=0.2))

    # upsample to 32X32
    g_model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    g_model.add(LeakyReLU(alpha=0.2))
    
    # upsample to 64X64
    g_model.add(Conv2DTranspose(64, (8,8), strides=(2,2), padding='same'))
    g_model.add(LeakyReLU(alpha=0.2))

    # upsample to 128X128
    g_model.add(Conv2DTranspose(32, (8,8), strides=(2,2), padding='same'))
    g_model.add(LeakyReLU(alpha=0.2))


    g_model.add(Conv2D(1, (5, 5), activation='tanh', padding='same'))
    
    return g_model
    
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)

    model.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    
    return model


def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space

    x_input = np.random.normal(0, 1, size=(n_samples, latent_dim))
    return x_input


def generate_fake_samples(g_model, latent_dim, batch_size):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, batch_size)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    #y = zeros((n_samples, 1))
    y = np.random.random_sample((batch_size,1)) * 0.2
    return X, y




def load_images():
    
    path="croped_pics/"
    X = (expand_dims(np.array([cv2.imread(p,0) for p in glob.glob(path+'*.jpg')]),axis=-1)-127.5)/127.5
    X = X.astype('float32')
    X = X[0:10000]
    return X

# select real samples
def generate_real_samples(dataset,index, batch_size):

    X = dataset[index * batch_size:(index + 1) * batch_size]

    # generate 'real' class labels (1)
    #y_2 = ones((batch_size, 1))
    y  = ones((batch_size,1)) - np.random.random_sample((batch_size,1)) * 0.2
    return X, y


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=156):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    
    
    
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset,j, half_batch)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            
            # train discriminator
            dis_loss_real = d_model.train_on_batch(X_real, y_real)
            dis_loss_fake = d_model.train_on_batch(X_fake, y_fake)

            d_loss = (dis_loss_real+dis_loss_fake)/2

            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            #y_gan = ones((n_batch, 1))
            y_gan = ones((n_batch,1)) - np.random.random_sample((n_batch,1)) * 0.2
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
        # evaluate the model performance, sometimes
        if (i+1) % 1 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=3):
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename = 'generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()


def random_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100,n_plots=9):
    # prepare real samples
    X_real, y_real = random_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # save plot
    save_plot(x_fake[0:n_plots], epoch)
    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    g_model.save(filename)


# size of the latent space
latent_dim = 300
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)    
# load image data
dataset = load_images()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

