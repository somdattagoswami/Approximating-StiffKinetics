import scipy.io as io
import numpy as np
import pickle
import sys

class DataSet_AE:
    def __init__(self, data_dir, data_type, bs):

        self.variables_folder = data_dir
        self.data_type = data_type
        self.bs = bs
        print(bs)
        # self.x_train, self.x_test, self.x_mean, self.x_std = self.load_data_ae()
        self.x_train, self.x_test, self.x_min, self.x_max = self.load_data_ae()

    def decode(self, x):
        # x = x*self.x_std + self.x_mean
        x = x*(self.x_max - self.x_min) + self.x_min
        return x  

    def load_data_ae(self):

        # Load dataset
        file_train = io.loadmat('./Data/training_data_ds')
        file_test = io.loadmat('./Data/testing_data_ds')

        # Prepare dataset
        # The equivalence ratio and the initial temperature are not considered for training the autoencoder.
        x_train = file_train['f_train'][:,0:12]
        data_250_train = file_train['u250_train']
        data_500_train = file_train['u500_train']
        data_750_train = file_train['u750_train']
        data_1000_train = file_train['u1000_train']
        x_train = np.concatenate((x_train, data_250_train, data_500_train, data_750_train, data_1000_train), axis = 0)
        x_test = file_test['f_test'][:,0:12]
         
        # x_mean = np.reshape(np.mean(x_train, axis = 0), (1,x_train.shape[-1]))
        # x_std = np.reshape(np.std(x_train, axis = 0), (1,x_train.shape[-1]))
        # x_train = ((x_train - x_mean)/x_std).astype(self.data_type)
        # x_test = ((x_test- x_mean)/x_std).astype(self.data_type)
        # save_dict = {'x_mean': x_mean, 'x_std': x_std}            
        # pickle.dump(save_dict, open(self.variables_folder+'Normalization_vals_ae.pkl', 'wb'))

        x_min = np.reshape(np.min(x_train, axis = 0), (1,x_train.shape[-1]))
        x_max = np.reshape(np.max(x_train, axis = 0), (1,x_train.shape[-1]))

        x_train = ((x_train - x_min)/(x_max - x_min)).astype(self.data_type)
        x_test = ((x_test - x_min)/(x_max - x_min)).astype(self.data_type)

        save_dict = {'x_min': x_min, 'x_max': x_max}            
        pickle.dump(save_dict, open(self.variables_folder + 'Normalization_vals_ae.pkl', 'wb'))  

        # return x_train, x_test, x_mean, x_std
        return x_train, x_test, x_min, x_max

    def minibatch(self):
        batch_id = np.random.choice(self.x_train.shape[0], self.bs, replace=False)
        x_train = [self.x_train[i:i+1] for i in batch_id]
        x_train = np.concatenate(x_train, axis=0)
        return x_train

    def testbatch(self, ntest):
        batch_id = np.random.choice(self.x_test.shape[0], ntest, replace=False)
        x_test = [self.x_test[i:i+1] for i in batch_id]
        x_test = np.concatenate(x_test, axis=0)
        return x_test

    def print(self):
        return self.x_test   

class DataSet_DON:

    def __init__(self, data_dir, data_type, bs, latent_dim):

        self.bs = bs
        self.variables_folder = data_dir
        self.data_type = data_type
        self.latent_dim = latent_dim
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_loc = self.load_data_don(data_dir)

    def denormalize_don_outputs(self, y):
        
        y = y*self.Y_std + self.Y_mean
        return y 
    
    def normalize_don_inputs(self, x):
        
        x = (x - self.X_mean)/self.X_std
        
        return x 

    def normalize_ae(self, x):
        
        ae_norm_data = pickle.load(open(self.variables_folder + 'Normalization_vals_ae.pkl', 'rb'))
        # x_ae_mean = ae_norm_data['x_mean']
        # x_ae_std = ae_norm_data['x_std']
        # x = x*x_ae_std + x_ae_mean

        x_ae_min = ae_norm_data['x_min']
        x_ae_max = ae_norm_data['x_max']
        x = (x - x_ae_min)/(x_ae_max - x_ae_min) 

        return x 
    
    def denormalize_ae(self, x):
    
        ae_norm_data = pickle.load(open(self.variables_folder+'Normalization_vals_ae.pkl', 'rb'))
        # x_ae_mean = ae_norm_data['x_mean']
        # x_ae_std = ae_norm_data['x_std']
        # x = x*x_ae_std + x_ae_mean
        x_ae_min = ae_norm_data['x_min']
        x_ae_max = ae_norm_data['x_max']
        x = x*(x_ae_max - x_ae_min) + x_ae_min
    
        return x 
    
    def load_data_don(self, data_dir):    
        
        # Load all data (original + reduced)
        file_ld_train = np.load(data_dir + 'data_d_{}_train.npz'.format(self.latent_dim))
        file_ld_test = np.load(data_dir + 'data_d_{}_test.npz'.format(self.latent_dim))

        X_func_train = file_ld_train['data_f_train']
        Y_train = file_ld_train['data_u_train']
        X_func_test = file_ld_test['data_f_test']
        Y_test = file_ld_test['data_u_test']
        # Normalizing the data
        # X_func_mean = np.reshape(np.mean(X_func_train, axis = 0), (1,X_func_train.shape[1],self.latent_dim+2))
        # X_func_std = np.reshape(np.std(X_func_train, axis = 0), (1,X_func_train.shape[1],self.latent_dim+2))
        # Y_mean = np.reshape(np.mean(Y_train, axis = 0), (1,Y_train.shape[1],self.latent_dim))
        # Y_std = np.reshape(np.std(Y_train, axis = 0), (1,Y_train.shape[1],self.latent_dim))

        # save_dict = {'X_func_mean': X_func_mean, 'X_func_std': X_func_std, 'Y_mean': Y_mean, 'Y_std': Y_std}            
        # pickle.dump(save_dict, open(self.variables_folder+'Normalization_vals_don.pkl', 'wb'))

        # X_func_train = ((X_func_train - X_func_mean)/X_func_std).astype(self.data_type)
        # X_func_test = ((X_func_test - X_func_mean)/X_func_std).astype(self.data_type)
        # X_func_print = ((X_func_print - X_func_mean)/X_func_std).astype(self.data_type)
        # Y_train = ((Y_train - Y_mean)/Y_std).astype(self.data_type)
        # Y_test = ((Y_test - Y_mean)/Y_std).astype(self.data_type)
        # Y_print = ((Y_print - Y_mean[:,3:4,:])/Y_std[:,3:4,:]).astype(self.data_type)

        X_loc = np.reshape(np.array([0.250, 0.500, 0.750, 1.0]),(4,1))

        return X_func_train, Y_train, X_func_test, Y_test, X_loc

    def minibatch(self):
        
        batch_id = np.random.choice(self.X_train.shape[0], self.bs, replace=False)
        X_train = [self.X_train[i:i+1] for i in batch_id]
        X_train = np.concatenate(X_train, axis=0)
        Y_train = [self.Y_train[i:i+1] for i in batch_id]
        Y_train = np.concatenate(Y_train, axis=0)

        return self.X_loc, X_train, Y_train

    def testbatch(self, batch_size):
        
        batch_id = np.random.choice(self.X_test.shape[0], batch_size, replace=False)
        X_test = [self.X_test[i:i+1] for i in batch_id]
        X_test = np.concatenate(X_test, axis=0)
        Y_test = [self.Y_test[i:i+1] for i in batch_id]
        Y_test = np.concatenate(Y_test, axis=0)

        return self.X_loc, X_test, Y_test    

    def integratedbatch(self): 
        
        file_test = io.loadmat('./Data/testing_data_ds')
        x_test = file_test['f_test'][:,0:12]
        x_test_ic = file_test['f_test'][:,12:14]
        x_test = self.normalize_ae(x_test).astype(self.data_type)
        y1000_test = file_test['u1000_test'].reshape(-1,1,x_test.shape[-1])
        
        return self.X_loc, x_test, x_test_ic, y1000_test