import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io
import sys
from sklearn.decomposition import PCA
np.random.seed(1234)

class DataSet:
    def __init__(self, bs, save_results_to):
        self.bs = bs
        self.results_folder = save_results_to
        self.F_train, self.U250_train, self.U500_train, self.U750_train, self.U1000_train, \
        self.u250_mean, self.u250_std, self.u500_mean, self.u500_std, self.u750_mean, self.u750_std, \
            self.u1000_mean, self.u1000_std = self.load_data()

    def decoder(self, u250, u500, u750, u1000):
        
        u250 = u250*(self.u250_std) + self.u250_mean
        u500 = u500*(self.u500_std) + self.u500_mean
        u750 = u750*(self.u750_std) + self.u750_mean
        u1000 = u1000*(self.u1000_std) + self.u1000_mean
        
        return u250, u500, u750, u1000

    def load_data(self):

        data = np.load('./Data/datasetPointwise_4timeSteps.npz')
        species_num = 12
        
        f_train = data['f_train']
        u250_train = data['u250_train']
        u500_train = data['u500_train']
        u750_train = data['u750_train']
        u1000_train = data['u1000_train']

        save_dict = {'f': f_train, 'u250': u250_train, 'u500': u500_train, 'u750': u750_train, 'u1000': u1000_train}
        io.savemat(self.results_folder+'/test_inputdata.mat', save_dict)

        f_train_mean = np.mean(np.reshape(f_train, (-1, species_num)), 0)
        f_train_std = np.std(np.reshape(f_train, (-1, species_num)), 0)
        u250_train_mean = np.mean(np.reshape(u250_train, (-1, species_num)), 0)
        u250_train_std = np.std(np.reshape(u250_train, (-1, species_num)), 0)
        u500_train_mean = np.mean(np.reshape(u500_train, (-1, species_num)), 0)
        u500_train_std = np.std(np.reshape(u500_train, (-1, species_num)), 0)
        u750_train_mean = np.mean(np.reshape(u750_train, (-1, species_num)), 0)
        u750_train_std = np.std(np.reshape(u750_train, (-1, species_num)), 0)
        u1000_train_mean = np.mean(np.reshape(u1000_train, (-1, species_num)), 0)
        u1000_train_std = np.std(np.reshape(u1000_train, (-1, species_num)), 0)
        
        f_train_mean = np.reshape(f_train_mean, (-1, 1, species_num))
        f_train_std = np.reshape(f_train_std, (-1, 1, species_num))
        u250_train_mean = np.reshape(u250_train_mean, (-1, species_num, 1))
        u250_train_std = np.reshape(u250_train_std, (-1, species_num, 1))
        u500_train_mean = np.reshape(u500_train_mean, (-1, species_num, 1))
        u500_train_std = np.reshape(u500_train_std, (-1, species_num, 1))
        u750_train_mean = np.reshape(u750_train_mean, (-1, species_num, 1))
        u750_train_std = np.reshape(u750_train_std, (-1, species_num, 1))
        u1000_train_mean = np.reshape(u1000_train_mean, (-1, species_num, 1))
        u1000_train_std = np.reshape(u1000_train_std, (-1, species_num, 1))
        
        F_train = np.reshape(f_train, (-1, 1, species_num))
        F_train = (F_train - f_train_mean)/(f_train_std)
        U250_train = np.reshape(u250_train, (-1, species_num, 1))
        U250_train = (U250_train - u250_train_mean)/u250_train_std
        U500_train = np.reshape(u500_train, (-1, species_num, 1))
        U500_train = (U500_train - u500_train_mean)/u500_train_std
        U750_train = np.reshape(u750_train, (-1, species_num, 1))
        U750_train = (U750_train - u750_train_mean)/u750_train_std
        U1000_train = np.reshape(u1000_train, (-1, species_num, 1))
        U1000_train = (U1000_train - u1000_train_mean)/u1000_train_std

        return F_train, U250_train, U500_train, U750_train, U1000_train, u250_train_mean, u250_train_std, u500_train_mean, u500_train_std,\
            u750_train_mean, u750_train_std, u1000_train_mean, u1000_train_std
        
    def minibatch(self):
        
        batch_id = np.random.choice(self.F_train.shape[0], self.bs, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u250_train = [self.U250_train[i:i+1] for i in batch_id]
        u250_train = np.concatenate(u250_train, axis=0)
        u500_train = [self.U500_train[i:i+1] for i in batch_id]
        u500_train = np.concatenate(u500_train, axis=0)
        u750_train = [self.U750_train[i:i+1] for i in batch_id]
        u750_train = np.concatenate(u750_train, axis=0)
        u1000_train = [self.U1000_train[i:i+1] for i in batch_id]
        u1000_train = np.concatenate(u1000_train, axis=0)
        
        x_train = np.reshape(np.array([0.250, 0.500, 0.750, 1.0]),(4,1))

        return x_train, f_train, u250_train, u500_train, u750_train, u1000_train

    def testbatch(self, num_test):
        
        batch_id = np.random.choice(self.F_train.shape[0], num_test, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u250_train = [self.U250_train[i:i+1] for i in batch_id]
        u250_train = np.concatenate(u250_train, axis=0)
        u500_train = [self.U500_train[i:i+1] for i in batch_id]
        u500_train = np.concatenate(u500_train, axis=0)
        u750_train = [self.U750_train[i:i+1] for i in batch_id]
        u750_train = np.concatenate(u750_train, axis=0)
        u1000_train = [self.U1000_train[i:i+1] for i in batch_id]
        u1000_train = np.concatenate(u1000_train, axis=0)
        
        x_train = np.reshape(np.array([0.250, 0.500, 0.750, 0.1000]),(4,1))

        return x_train, f_train, u250_train, u500_train, u750_train, u1000_train

    def testbatch_print(self, num_test):
        
        batch_id = np.arange(num_test)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u250_train = [self.U250_train[i:i+1] for i in batch_id]
        u250_train = np.concatenate(u250_train, axis=0)
        u500_train = [self.U500_train[i:i+1] for i in batch_id]
        u500_train = np.concatenate(u500_train, axis=0)
        u750_train = [self.U750_train[i:i+1] for i in batch_id]
        u750_train = np.concatenate(u750_train, axis=0)
        u1000_train = [self.U1000_train[i:i+1] for i in batch_id]
        u1000_train = np.concatenate(u1000_train, axis=0)
        
        x_train = np.reshape(np.array([0.250, 0.500, 0.750, 0.1000]),(4,1))

        return x_train, f_train, u250_train, u500_train, u750_train, u1000_train
