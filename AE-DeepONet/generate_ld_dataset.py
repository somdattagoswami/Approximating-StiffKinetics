import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import time, random, os, sys
import matplotlib
import matplotlib.pyplot as plt
from AE import AE
import scipy.io as io
import argparse
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress warnings

#### Parser
parser = argparse.ArgumentParser(description='Running autoencoder models.')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=6,
    help='latent dimensionality (default: 6)')
parser.add_argument(
    '--bs',
    type=int,
    default=5000,
    help='batch size (default: 5000)')  
parser.add_argument(
    '--dtype',
    type=str,
    default="float64",
    help='data type (default = float64)') 

args, unknown = parser.parse_known_args()

latent_dim = args.latent_dim
batch_size_train = 5000 # Change this according to your dataset size.
batch_size_test = 500 # Change this according to your dataset size. 
data_type = args.dtype
data_dir = 'results/d_' + str(args.latent_dim) + '/data/'
class_AE_dir = 'results/d_' + str(args.latent_dim) + '/class_AE/'

# Create directories for results
if not os.path.exists('results/d_' + str(args.latent_dim) + '/data/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/data/')

def encode_decode(model, x, fname, make_fig=False):
    ld = model.encode(x)
    print('low dimensional data: ', ld.shape)

    return(ld)

def normalize_ae(x):
        
    ae_norm_data = pickle.load(open(data_dir + 'Normalization_vals_ae.pkl', 'rb'))
    
    # x_ae_mean = ae_norm_data['x_mean']
    # x_ae_std = ae_norm_data['x_std']
    # x = x*x_ae_std + x_ae_mean

    x_ae_min = ae_norm_data['x_min']
    x_ae_max = ae_norm_data['x_max']
    x = (x - x_ae_min)/(x_ae_max - x_ae_min) 

    return x 

def main():
    
    # Load dataset
    file = io.loadmat('./Data/training_data_ds')
    data_f_train = file['f_train'][:,0:12] # Removing initial temperature, initial equivalence ratio 
    data_ic_train = file['f_train'][:,12:14].reshape(-1,1,2)
    data_250_train = file['u250_train']
    data_500_train = file['u250_train']
    data_750_train = file['u750_train']
    data_1000_train = file['u1000_train']

    file = io.loadmat('./Data/testing_data_ds')
    data_f_test = file['f_test'][:,0:12]
    data_ic_test = file['f_test'][:,12:14].reshape(-1,1,2)
    data_250_test = file['u250_test']
    data_500_test = file['u250_test']
    data_750_test = file['u750_test']
    data_1000_test = file['u1000_test']

    data_f_train = normalize_ae(data_f_train).astype(data_type)
    data_250_train = normalize_ae(data_250_train).astype(data_type)
    data_500_train = normalize_ae(data_500_train).astype(data_type)
    data_750_train = normalize_ae(data_750_train).astype(data_type)
    data_1000_train = normalize_ae(data_1000_train).astype(data_type)

    data_f_test = normalize_ae(data_f_test).astype(data_type)
    data_250_test = normalize_ae(data_250_test).astype(data_type)
    data_500_test = normalize_ae(data_500_test).astype(data_type)
    data_750_test = normalize_ae(data_750_test).astype(data_type)
    data_1000_test = normalize_ae(data_1000_test).astype(data_type)

    num_species = data_f_train.shape[-1]
    
    # Loading model
    model = AE(latent_dim, num_species, data_type)
    model_number = np.load(class_AE_dir+'Best_AE_model_number.npy')
    model_address = class_AE_dir + "model_"+ str(model_number)
    model.load_weights(model_address)

    train_f_ld_ls = []
    train250_ld_ls = []
    train500_ld_ls = []
    train750_ld_ls = []
    train1000_ld_ls = []
    test_f_ld_ls = []
    test250_ld_ls = []
    test500_ld_ls = []
    test750_ld_ls = []
    test1000_ld_ls = []
    print_f_ld_ls = []
    print1000_ld_ls = []


    for end in np.arange(batch_size_train, data_f_train.shape[0]+1, batch_size_train):
        start=end-batch_size_train
        train_f_ld_ls.append(encode_decode(model, data_f_train[start:end], 'train'))
        print('end: ', end)

    for end in np.arange(batch_size_train, data_250_train.shape[0]+1, batch_size_train):
        start=end-batch_size_train
        train250_ld_ls.append(encode_decode(model, data_250_train[start:end], 'train'))
        print('end: ', end)

    for end in np.arange(batch_size_train, data_500_train.shape[0]+1, batch_size_train):
        start=end-batch_size_train
        train500_ld_ls.append(encode_decode(model, data_500_train[start:end], 'train'))
        print('end: ', end)

    for end in np.arange(batch_size_train, data_750_train.shape[0]+1, batch_size_train):
        start=end-batch_size_train
        train750_ld_ls.append(encode_decode(model, data_750_train[start:end], 'train'))
        print('end: ', end)

    for end in np.arange(batch_size_train, data_1000_train.shape[0]+1, batch_size_train):
        start=end-batch_size_train
        train1000_ld_ls.append(encode_decode(model, data_1000_train[start:end], 'train'))
        print('end: ', end)

    for end in np.arange(batch_size_test, data_f_test.shape[0]+1, batch_size_test):
        start=end-batch_size_test
        test_f_ld_ls.append(encode_decode(model, data_f_test[start:end], 'test'))
        print('end: ', end)

    for end in np.arange(batch_size_test, data_250_test.shape[0]+1, batch_size_test):
        start=end-batch_size_test
        test250_ld_ls.append(encode_decode(model, data_250_test[start:end], 'test'))
        print('end: ', end)

    for end in np.arange(batch_size_test, data_500_test.shape[0]+1, batch_size_test):
        start=end-batch_size_test
        test500_ld_ls.append(encode_decode(model, data_500_test[start:end], 'test'))
        print('end: ', end)

    for end in np.arange(batch_size_test, data_750_test.shape[0]+1, batch_size_test):
        start=end-batch_size_test
        test750_ld_ls.append(encode_decode(model, data_750_test[start:end], 'test'))
        print('end: ', end)

    for end in np.arange(batch_size_test, data_1000_test.shape[0]+1, batch_size_test):
        start=end-batch_size_test
        test1000_ld_ls.append(encode_decode(model, data_1000_test[start:end], 'test'))
        print('end: ', end)

    train_f_ld = np.concatenate(train_f_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    train_f_ld = np.concatenate((train_f_ld,data_ic_train),axis = -1) # Integrating the initial conditions
    train250_ld = np.concatenate(train250_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    train500_ld = np.concatenate(train500_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    train750_ld = np.concatenate(train750_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    train1000_ld = np.concatenate(train1000_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    train_u_ld = np.concatenate((train250_ld, train500_ld, train750_ld, train1000_ld), axis=1)
    del train250_ld, train500_ld, train750_ld, train1000_ld

    test_f_ld = np.concatenate(test_f_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    test_f_ld = np.concatenate((test_f_ld,data_ic_test),axis = -1)
    test250_ld = np.concatenate(test250_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    test500_ld = np.concatenate(test500_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    test750_ld = np.concatenate(test750_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    test1000_ld = np.concatenate(test1000_ld_ls,axis=0).reshape(-1,1,args.latent_dim)
    test_u_ld = np.concatenate((test250_ld, test500_ld, test750_ld, test1000_ld), axis=1)
    del test250_ld, test500_ld, test750_ld, test1000_ld
    

    # Save reduced data (for DeepONet)
    np.savez(data_dir + 'data_d_{}_train.npz'.format(args.latent_dim), latent_dim=args.latent_dim, 
                                                    data_f_train=train_f_ld, data_u_train=train_u_ld)  

    np.savez(data_dir + 'data_d_{}_test.npz'.format(args.latent_dim), latent_dim=args.latent_dim, 
                                                    data_f_test=test_f_ld, data_u_test=test_u_ld) 

    print('Prepared the data for training and testing DeepONet.')

if __name__ == '__main__':
    main()