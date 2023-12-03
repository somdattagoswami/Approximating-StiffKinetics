import scipy.io as io
import numpy as np
import pickle
import sys

class DataSet_AE:
    def __init__(self, data_dir, data_type, bs):

        self.variables_folder = data_dir
        self.data_type = data_type
        self.bs = bs
        # self.x_train, self.x_test, self.x_mean, self.x_std = self.load_data_ae()
        self.x_train, self.x_test, self.x_min, self.x_max = self.load_data_ae()

    def decode(self, x):
        # x = x*self.x_std + self.x_mean
        x = x*(self.x_max - self.x_min) + self.x_min
        return x  

    def load_data_ae(self):

        # Load dataset
        file = io.loadmat('../Data/DataLog_ae_training')
        data = file['Y1']

        # Prepare dataset
        shuffler = np.random.permutation(len(data))

        num_train = int(0.9*len(data))
        num_test = len(data) - num_train
        x_train = data[shuffler[0:num_train],:] 
        x_test = data[shuffler[num_train::],:]

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

    def testing(self):

        # Load dataset
        file = io.loadmat('../Data/printing_data_temp_eqr')
        x_test = file['f_print'][:,2:14] # Removing initial temperature, initial equivalence ratio
        x_test = np.log(x_test+1e-10)
        # x_test = ((x_test - self.x_mean)/self.x_std).astype(self.data_type)
        x_test = ((x_test - self.x_min)/(self.x_max - self.x_min)).astype(self.data_type)

        return x_test   

class DataSet_DON:

    def __init__(self, data_dir, data_type, bs, latent_dim):

        self.bs = bs
        self.variables_folder = data_dir
        self.data_type = data_type
        self.latent_dim = latent_dim
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.X_print, \
            self.Y_print, self.X_loc, self.X_mean, self.Y_mean, self.X_std, \
            self.Y_std = self.load_data_don(data_dir)

    def denormalize_don(self, y):
        
        y = y*self.Y_std + self.Y_mean
        return y 
    
    def normalize_don_inputs(self, x):
        
        x = (x - self.X_mean)/self.X_std
        
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

    def normalize_ae(self, x):
        
        ae_norm_data = pickle.load(open(self.variables_folder + 'Normalization_vals_ae.pkl', 'rb'))
        # x_ae_mean = ae_norm_data['x_mean']
        # x_ae_std = ae_norm_data['x_std']
        # x = x*x_ae_std + x_ae_mean

        x_ae_min = ae_norm_data['x_min']
        x_ae_max = ae_norm_data['x_max']
        x = (x - x_ae_min)/(x_ae_max - x_ae_min) 

        return x 

    def load_data_don(self, data_dir):    
        
        # Load all data (original + reduced)
        # The ld data is on normalized values
        file_ld_train = np.load(data_dir + 'data_d_{}_train.npz'.format(self.latent_dim))
        file_ld_test = np.load(data_dir + 'data_d_{}_test.npz'.format(self.latent_dim))
        file_ld_print = np.load(data_dir + 'data_d_{}_print.npz'.format(self.latent_dim))

        file_og_train = io.loadmat('../Data/training_data_temp_eqr_log')
        file_og_test = io.loadmat('../Data/testing_data_temp_eqr_log')
        file_og_print = io.loadmat('../Data/printing_data_temp_eqr_log')

        #Tfac = 1000
        temp = file_og_train['f_train'][:,0:2].reshape(-1,1,2)#/[Tfac,1]
        X_func_train = np.concatenate((temp, file_ld_train['data_f_train']),axis = -1)
        del file_og_train
        Y_train = file_ld_train['data_u_train']

        temp = file_og_test['f_test'][:,0:2].reshape(-1,1,2)#/[Tfac,1]
        X_func_test = np.concatenate((temp, file_ld_test['data_f_test']),axis = -1).astype(self.data_type)
        del file_og_test
        Y_test = file_ld_test['data_u_test']

        temp = file_og_print['f_print'][:,0:2].reshape(-1,1,2)#/[Tfac,1]
        X_func_print = np.concatenate((temp, file_ld_print['data_f_print']),axis = -1)
        del file_og_print, temp
        Y_print = file_ld_print['data_1000_print']

        # Normalizing the data
        X_func_mean = np.reshape(np.mean(X_func_train, axis = 0), (1,X_func_train.shape[1],self.latent_dim+2))
        X_func_std = np.reshape(np.std(X_func_train, axis = 0), (1,X_func_train.shape[1],self.latent_dim+2))
        Y_mean = np.reshape(np.mean(Y_train, axis = 0), (1,Y_train.shape[1],self.latent_dim))
        Y_std = np.reshape(np.std(Y_train, axis = 0), (1,Y_train.shape[1],self.latent_dim))

        save_dict = {'X_func_mean': X_func_mean, 'X_func_std': X_func_std, 'Y_mean': Y_mean, 'Y_std': Y_std}            
        pickle.dump(save_dict, open(self.variables_folder+'Normalization_vals_don.pkl', 'wb'))

        X_func_train = ((X_func_train - X_func_mean)/X_func_std).astype(self.data_type)
        X_func_test = ((X_func_test - X_func_mean)/X_func_std).astype(self.data_type)
        X_func_print = ((X_func_print - X_func_mean)/X_func_std).astype(self.data_type)
        Y_train = ((Y_train - Y_mean)/Y_std).astype(self.data_type)
        Y_test = ((Y_test - Y_mean)/Y_std).astype(self.data_type)
        Y_print = ((Y_print - Y_mean[:,3:4,:])/Y_std[:,3:4,:]).astype(self.data_type)

        X_loc = np.reshape(np.array([0.250, 0.500, 0.750, 1.0]),(4,1))

        return X_func_train, Y_train, X_func_test, Y_test, X_func_print, \
            Y_print, X_loc, X_func_mean, Y_mean, X_func_std, Y_std

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

    def integrated_testbatch(self, ntest): 
        
        file_og_test = io.loadmat('../Data/testing_data_temp_eqr_log')
        X_test = file_og_test['f_test'][:ntest,2:14]
        X_init = file_og_test['f_test'][:ntest,0:2]
        X_test = self.normalize_ae(X_test).astype(self.data_type)

        u250_true = file_og_test['u250_test'][:ntest, :].reshape(-1,1,X_test.shape[-1])
        u500_true = file_og_test['u500_test'][:ntest, :].reshape(-1,1,X_test.shape[-1])
        u750_true = file_og_test['u750_test'][:ntest, :].reshape(-1,1,X_test.shape[-1])
        u1000_true = file_og_test['u1000_test'][:ntest, :].reshape(-1,1,X_test.shape[-1])
        Y_test = np.concatenate((u250_true, u500_true, u750_true, u1000_true), axis = 1)
        
        return self.X_loc, X_test, Y_test, X_init

    def printbatch(self): 
        
        file_og_print = io.loadmat('../Data/printing_data_temp_eqr_log')
        X_test = file_og_print['f_print'][:,2:14]
        X_init = file_og_print['f_print'][:,0:2]
        X_test = self.normalize_ae(X_test).astype(self.data_type)

        u1000_true = file_og_print['u1000_print'].reshape(-1,1,X_test.shape[-1])
        Y_test = u1000_true
        
        return self.X_loc, X_test, Y_test, X_init