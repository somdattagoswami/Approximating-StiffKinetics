import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input, initializers
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D,  PReLU, Flatten, Dense, Activation, LeakyReLU
import matplotlib
import matplotlib.pyplot as plt
import time
import os

class AE(tf.keras.Model):

    def __init__(self, latent_dim, num_species, data_type):
        tf.keras.mixed_precision.set_global_policy(data_type)
        super(AE, self).__init__()

        # Defining some model parameters
        self.latent_dim = latent_dim
        self.num_species = num_species
        self.index_list = [] 
        self.train_loss_list = []
        self.val_loss_list = []

        self.lr=10**-4

        self.encoder_ls = self.build_encoder()
        self.decoder_ls  = self.build_decoder()

    def build_encoder(self):
        
        ls=[]
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(self.latent_dim,name='dense1'))

        return ls

    def build_decoder(self):
        
        ls=[]
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(self.num_species, Activation(LeakyReLU()),GlorotNormal()))
        
        return ls

    @tf.function(jit_compile=True)
    def call(self, x):
        y=x
        for i in range(len(self.encoder_ls)):
            y = self.encoder_ls[i](y)

        for i in range(len(self.decoder_ls)):
            y = self.decoder_ls[i](y)

        return y

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):
        train_loss = tf.reduce_mean(tf.square(y_train - y_pred))

        return([train_loss])

    @tf.function(jit_compile=True)
    def encode(self, x):
        y = x
        for i in range(len(self.encoder_ls)):
            y = self.encoder_ls[i](y)

        return y

    @tf.function(jit_compile=True)
    def decode(self, x):
        y=x
        for i in range(len(self.decoder_ls)):
            y = self.decoder_ls[i](y)

        return y