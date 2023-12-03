import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
import time, random, os, sys
import matplotlib.pyplot as plt
import scipy.io as io
from AE import AE
import argparse
import pickle
from dataset import DataSet_AE
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # supress warnings

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
    default = 800,
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

@tf.function(jit_compile=True)
def train(model, x, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss   = model.loss(y_pred, x)[0]

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return(loss)

def tensor(x):
    return tf.convert_to_tensor(x, dtype=data_type)

# Create directories for results
if not os.path.exists('results/d_' + str(args.latent_dim) + '/plots/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + '/plots/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/errors_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/' + '/errors_AE/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/class_AE/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/class_AE/')    
if not os.path.exists('results/d_' + str(args.latent_dim) + '/data/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/data/')
if not os.path.exists('results/d_' + str(args.latent_dim) + '/pred/'):
    os.makedirs('results/d_' + str(args.latent_dim) + '/pred/') 

class_AE_dir = 'results/d_' + str(args.latent_dim) + '/class_AE/'
data_dir = 'results/d_' + str(args.latent_dim) + '/data/'

# Fix random see (for reproducibility of results)
seed_value = random.randint(1,1000)
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

n_epochs = args.n_epochs
batch_size = args.bs
latent_dim = args.latent_dim
data_type = args.dtype
savestep = 100

data = DataSet_AE(data_dir, data_type, batch_size)
def main():
    
    num_species = 12

    # Create an object
    model = AE(latent_dim, num_species, data_type)
    print('Autoencoder model is created.')

    optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-4)

    # Run autoencoder
    print('Seed number:', seed_value)
    print('Autoencoder training in progress...')

    begin_time = time.time()
    for i in range(n_epochs+1):
        x_train = data.minibatch()
        loss = train(model, tensor(x_train), optimizer)

        if i % savestep == 0:

            model.save_weights(class_AE_dir + "model_"+str(i))
            train_loss = loss.numpy()
            x_test = data.testbatch(ntest=50000)
            y_pred = model(x_test)
            # val_loss = np.mean((y_pred - x_test)**2)
            val_loss = np.mean(np.linalg.norm(y_pred - x_test, 2, axis=1)/np.linalg.norm(x_test, axis=1))
            print("epoch:" + str(i) + ", Train Loss:" + "{:.3e}".format(train_loss) + ", Val Loss:" + "{:.3e}".format(val_loss) +  ", elapsed time: " +  str(int(time.time()-begin_time)) + "s"  )

            model.index_list.append(i)
            model.train_loss_list.append(train_loss)
            model.val_loss_list.append(val_loss)

    print('Autoencoder training is completed.')

    # Convergence plot
    index_list = model.index_list
    train_loss_list = model.train_loss_list
    val_loss_list = model.val_loss_list
    errdir = 'results/d_' + str(args.latent_dim) + '/errors_AE/'
    np.savez(errdir +'convergence_data', index_list=index_list, train_loss_list=train_loss_list, val_loss_list=val_loss_list)

    plt.close()
    fig = plt.figure(figsize=(4,3))
    plt.plot(index_list, train_loss_list, label="train", linewidth=2)
    plt.plot(index_list, val_loss_list, label="validation", linewidth=2)
    plt.title('Autoencoder loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.yscale('log')
    plt.tight_layout()
    plotdir = 'results/d_' + str(args.latent_dim) + '/plots/'
    plt.savefig(plotdir + 'History_AE_seed_' + str(seed_value) + '.png', dpi=300)

    # Compute L2 error on print data
    x_test = data.testing()
    encoded_test = model.encode(x_test).numpy()
    decoded_test = model(x_test).numpy()
    reference_data = data.decode(x_test)
    decoded_test = data.decode(decoded_test)
    errors = decoded_test- reference_data
    l2_rel_err = np.mean(np.linalg.norm(errors, 2, axis=1)/np.linalg.norm(reference_data, axis=1))
    print('Autoencoder relative L2 error: {}\n'.format(round(l2_rel_err,4)))
    errordir = 'results/d_' + str(args.latent_dim) + '/errors_AE/'
    np.savetxt(errordir + 'error_' + '_seed_' + str(seed_value) + '.txt', 
                                np.expand_dims(np.array(l2_rel_err), axis=0), fmt='%e') 
    pred_dir = 'results/d_' + str(args.latent_dim) + '/pred/'
    save_dict = {'ref': reference_data, 'pred': decoded_test, 'ld': encoded_test, 'error': l2_rel_err}
    io.savemat(pred_dir + 'AE_results.mat', save_dict)  

    best_model_number = index_list[np.argmin(val_loss_list)]
    print('Best autencoder model: ', best_model_number)

    np.save(class_AE_dir+'Best_AE_model_number', best_model_number)
    
if __name__ == '__main__':
    main()
