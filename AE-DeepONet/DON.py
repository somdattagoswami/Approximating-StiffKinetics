import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Reshape, Conv2D, PReLU, Flatten, \
    Dense, Activation, MaxPooling2D, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.losses import MeanSquaredError
import matplotlib
import matplotlib.pyplot as plt
import time

import os

class DeepONet_Model(tf.keras.Model):
    
    def __init__(self, Par, latent_dim, data_type):
        tf.keras.mixed_precision.set_global_policy(data_type)
        super(DeepONet_Model, self).__init__()

        # Defining model parameters
        self.p = 10 # p nodes per component of the latent dimension
        self.m = latent_dim 
        self.Par = Par
        
        self.index_list = []
        self.train_loss_list = []
        self.val_loss_list = []
        self.branch_net_ls = self.build_branch_net()
        self.trunk_net_ls  = self.build_trunk_net()

        # self.alpha = tf.Variable(1, trainable=True)

    def build_branch_net(self):
        
        ls=[]
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(BatchNormalization())
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(BatchNormalization())
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(BatchNormalization())
        ls.append(Dense(self.m*self.p, name='dense1'))

        return ls

    def build_trunk_net(self):
        
        ls=[]
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(128, Activation(LeakyReLU()),GlorotNormal()))
        ls.append(Dense(self.m*self.p, name='dense2'))

        return ls

    @tf.function(jit_compile=True)
    def call(self, X_func, X_loc):

        y_func = X_func
        y_loc = X_loc

        for i in range(len(self.branch_net_ls)):
            y_func = self.branch_net_ls[i](y_func)

        for i in range(len(self.trunk_net_ls)):
            y_loc = self.trunk_net_ls[i](y_loc)

        y_func = tf.reshape(y_func, [-1, self.m, self.p])
        y_loc = tf.reshape(y_loc, [-1, self.m, self.p])

        Y = tf.einsum('ijk,pjk->ipj', y_func, y_loc)

        return(Y)

    @tf.function(jit_compile=True)
    def loss(self, y_pred, y_train):

        train_loss =  tf.reduce_mean(tf.square(y_pred - y_train))
        # train_loss = tf.reduce_mean(tf.norm(y_train - y_pred, 2, axis=1)/tf.norm(y_train, 2, axis=1))
        # train_loss = tf.reduce_sum(tf.norm(y_train - y_pred, 2, axis=1)/tf.norm(y_train, 2, axis=1))

        return([train_loss])