import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from AE import AE
from DON import DeepONet_Model
import matplotlib
import scipy.io as io
import argparse
import random
import os, sys, time
from dataset import DataSet_DON
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # supress warnings

#### Parser
parser = argparse.ArgumentParser(description='Running autoencoder models.')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=6,
    help='latent dimensionality (default: 6)')
parser.add_argument(
    '--n_epochs',
    type=int,
    default=800,
    help='number of epochs (default: 800)')
parser.add_argument(
    '--bs',
    type=int,
    default=50000,
    help='batch size (default: 50000)')
parser.add_argument(
    '--dtype',
    type=str,
    default="float64",
    help='data type (default = float64)') 

args, unknown = parser.parse_known_args()

#### Fix random see (for reproducibility of results)
seed_value = random.randint(1,1000)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

if not os.path.exists('results/d_' + str(args.latent_dim) + '/errors_DON/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/errors_DON/')     
if not os.path.exists('results/d_' + str(args.latent_dim) + '/class_DON/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/class_DON/') 
if not os.path.exists('results/d_' + str(args.latent_dim) + '/pred/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/pred/') 

n_epochs = args.n_epochs
batch_size = args.bs
latent_dim = args.latent_dim
data_type = args.dtype
Flag = 1
savestep = 100
data_dir = 'results/d_' + str(args.latent_dim) + '/data/'
data = DataSet_DON(data_dir, data_type, batch_size, latent_dim)

def tensor(x):
    return tf.convert_to_tensor(x, dtype=data_type)

@tf.function(jit_compile=True)
def train(don_model, optimizer, X_func, X_loc, y):
    with tf.GradientTape() as tape:
        y_hat  = don_model(X_func, X_loc)
        loss   = don_model.loss(y_hat, y)[0]

    gradients = tape.gradient(loss, don_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, don_model.trainable_variables))
    return(loss)

def error_metric(true, pred):

    nt = 4
    pred = np.reshape(pred, (-1,nt,pred.shape[-1]))
    num = np.abs(true - pred)**2
    num = np.sum(num) 
    den = np.abs(true)**2
    den = np.sum(den)

    return num/den

def show_error(don_model, ae_model, X_func, X_loc, u_true, data, save, class_dir, name):
    
    y_pred = don_model(X_func, X_loc)
    y_pred = data.denormalize_don(y_pred)
    y_pred = np.reshape(y_pred, (-1,args.latent_dim))
    y_pred = ae_model.decode(y_pred)
    y_pred = data.denormalize_ae(y_pred)
    error = error_metric(u_true, y_pred)
    print('L2 norm of relative error: ', error)

    if save==True:
        save_dict = {'ref': u_true, 'pred': y_pred, 'error': error}
        io.savemat(class_dir + name + '_results.mat', save_dict)  

    return error

def encode_decode(model, x, fname, make_fig=False):
    
    ld = model.encode(x)
    print('low dimensional data: ', ld.shape)

    return(ld)

def main():
    
    Par = {}
    class_DON_dir = 'results/d_' + str(args.latent_dim) + '/class_DON/'
    Par['address'] = class_DON_dir

    don_model = DeepONet_Model(Par, latent_dim, data_type)
    n_epochs = args.n_epochs
    batch_size = args.bs
    optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-4)
    print('DeepONet training in progress...')    
    
    begin_time = time.time()

    for i in range(n_epochs+1):
        X_loc, X_train, Y_train = data.minibatch()
        loss = train(don_model, optimizer, tensor(X_train), tensor(X_loc), tensor(Y_train))

        if i%savestep == 0:

            don_model.save_weights(Par['address'] + "/model_"+str(i))
            train_loss = loss.numpy()
            X_loc, X_test, Y_test = data.testbatch(batch_size)
            y_hat = don_model(X_test, X_loc)

            err250 = np.mean(np.linalg.norm(y_hat[:,0:1,:] - Y_test[:,0:1,:], 2, axis=1)/np.linalg.norm(Y_test[:,0:1,:], 2, axis=1))
            err500 = np.mean(np.linalg.norm(y_hat[:,1:2,:] - Y_test[:,1:2,:], 2, axis=1)/np.linalg.norm(Y_test[:,1:2,:], 2, axis=1))
            err750 = np.mean(np.linalg.norm(y_hat[:,2:3,:] - Y_test[:,2:3,:], 2, axis=1)/np.linalg.norm(Y_test[:,2:3,:], 2, axis=1))
            err1000 = np.mean(np.linalg.norm(y_hat[:,3:4,:] - Y_test[:,3:4,:], 2, axis=1)/np.linalg.norm(Y_test[:,3:4,:], 2, axis=1))

            print('Epoch: %d, Loss: %.3e, Error 250: %.3f, 500: %.3f, 750: %.3f, 1000: %.3f, Time (s): %.3f'%(i, train_loss, err250, err500, err750, err1000, int(time.time()-begin_time)))
            val_loss = err250 + err500 + err750 + err1000

            don_model.index_list.append(i)
            don_model.train_loss_list.append(train_loss)
            don_model.val_loss_list.append(val_loss)

    # Convergence plot
    index_list = don_model.index_list
    train_loss_list = don_model.train_loss_list
    val_loss_list = don_model.val_loss_list
    errordir = 'results/d_' + str(args.latent_dim) + '/errors_DON/'
    np.savez(errordir +'convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)
    
    plt.close()
    plt.figure(figsize=(4,3))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="validation", linewidth=2)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.yscale('log')
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title('Latent DeepONet loss')
    plt.tight_layout()
    plotdir = 'results/d_' + str(args.latent_dim) + '/plots/'
    plt.savefig(plotdir + 'History_DON_seed_' + str(seed_value) + '.png', dpi=300)
    plt.close()

    print('DeepONet training is completed.')
    class_AE_dir = 'results/d_' + str(args.latent_dim) + '/class_AE/'
    num_species = 12

    if Flag == True:

        # Loading model
        ae_model = AE(latent_dim, num_species, data_type)
        ae_model_number = np.load(class_AE_dir+'Best_AE_model_number.npy')
        ae_model_address = class_AE_dir + "model_"+str(ae_model_number)
        ae_model.load_weights(ae_model_address)

        don_model = DeepONet_Model(Par, latent_dim, data_type)
        don_model_number = index_list[np.argmin(val_loss_list)]
        np.save(Par['address'] + 'best_don_model_number', don_model_number)
        don_model_address = Par['address'] + "/model_"+str(don_model_number)
        don_model.load_weights(don_model_address)

        print('Best DeepONet model: ', don_model_number)
        
        # Checking the autoencoder + deeponet model on the test dataset
        print('')
        print('Test Dataset')

        X_loc, X_test, Y_test, X_test_init = data.integrated_testbatch(ntest=50000)
        X_test = encode_decode(ae_model, X_test, 'test').numpy()
        X_test = np.concatenate((X_test_init, X_test), axis=-1)
        X_test = data.normalize_don_inputs(X_test).astype(data_type)
        pred_dir = 'results/d_' + str(args.latent_dim) + '/pred/'
        error_test = show_error(don_model, ae_model, X_test, X_loc, Y_test, data, \
            save=True, class_dir=pred_dir, name = 'test')

        # Checking the model on the print dataset
        print('')
        print('Print Dataset')
        X_loc, X_print, Y_print, X_print_init = data.printbatch()
        X_print = encode_decode(ae_model, X_print, 'print').numpy()
        X_print = np.concatenate((X_print_init, X_print), axis=-1)
        X_print = data.normalize_don_inputs(X_print).astype(data_type)    
        y_pred = don_model(X_print, X_loc)
        y_pred = data.denormalize_don(y_pred)
        y_pred = np.reshape(y_pred, (-1,args.latent_dim))
        y_pred = ae_model.decode(y_pred)
        y_pred = data.denormalize_ae(y_pred)
        y_pred = np.reshape(y_pred, (-1,4,y_pred.shape[-1]))
        error = np.sum(np.abs(Y_print - y_pred[:,3:4,:])**2)/np.sum(np.abs(Y_print)**2)
        save_dict = {'ref': Y_print, 'pred': y_pred[:,3:4,:], 'error': error}
        io.savemat(pred_dir + 'print_results.mat', save_dict)   

        print('L2 norm of relative error: ', error)
        np.savetxt(errordir + 'error_DON_seed_' + str(seed_value) + '.txt', 
                                   np.expand_dims(np.array(error_test), axis=0), fmt='%e') 
        print('--------Complete--------')


if __name__ == '__main__':
    main()
