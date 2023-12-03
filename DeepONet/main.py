import os
#import tensorflow.compat.v1 as tf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from dataset import DataSet
from fnn import FNN
from savedata import SaveData

np.random.seed(1234)
#tf.set_random_seed(1234)

#output dimension of Branch/Trunk
p = 12*4
num = 12
#branch net
layer_B = [num, 128, 128, 128, p]
#trunk net
layer_T = [1, 128, 128, 128, p]
#resolution
h = num
#batch_size
bs = 1000
bs_test = 1000
#size of i
x_num = 4
epochs = 550001

num_test = 691*100
num_noisy = 100

save_index = 4
current_directory = os.getcwd()    
case = "Case_"
folder_index = str(save_index)
results_dir = "/" + case + folder_index +"/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)    
    
def main():
    
    data = DataSet(bs, save_results_to)
    x_train, f_train, u250_train, u500_train, u750_train, u1000_train = data.minibatch()

    f_ph = tf.placeholder(shape=[None, 1, num], dtype=tf.float32) #[bs, f_dim]
    u250_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    u500_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    u750_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    u1000_ph = tf.placeholder(shape=[None, num, 1], dtype=tf.float32) #[bs, x_num, 1]
    
    x_ph = tf.placeholder(shape=[x_num, 1], dtype=tf.float32)
    learning_rate = tf.placeholder(tf.float32, shape=[])

    fnn_model = FNN()
    # Branch net    
    W_B, b_B = fnn_model.hyper_initial(layer_B)
    u_B = fnn_model.fnn_B(W_B, b_B, f_ph)    
    
    #Trunk net
    W_T, b_T = fnn_model.hyper_initial(layer_T)
    u_T = fnn_model.fnn(W_T, b_T, x_ph)

    #inner product
    u_pred = tf.einsum('ijk,mk->ikm',u_B,u_T)

    u250_pred = u_pred[:,0:12,:]  
    u500_pred = u_pred[:,12:24,:]
    u750_pred = u_pred[:,24:36,:]
    u1000_pred = u_pred[:,36:48,:]
    
    u250_pred = tf.reduce_sum(u250_pred, axis=-1, keepdims=True)
    u500_pred = tf.reduce_sum(u500_pred, axis=-1, keepdims=True)
    u750_pred = tf.reduce_sum(u750_pred, axis=-1, keepdims=True)
    u1000_pred = tf.reduce_sum(u1000_pred, axis=-1, keepdims=True)
    
    loss = tf.reduce_mean(tf.square(u250_ph - u250_pred)) + tf.reduce_mean(tf.square(u500_ph - u500_pred)) + tf.reduce_mean(tf.square(u750_ph - u750_pred)) +\
        tf.reduce_mean(tf.square(u1000_ph - u1000_pred))
    # loss = tf.reduce_sum(tf.norm(u_pred - u_ph, 2, axis=1)/tf.norm(u_ph, 2, axis=1))
    train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))  
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nt = 0
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    
    train_loss = np.zeros((epochs+1, 1))
    test_loss = np.zeros((int(epochs/100)+1, 4)) 
    
    while n <= epochs:
        
        if n < 50000:
            lr = 0.001
        elif n < 100000:
            lr = 0.0005
        else:
            lr = 0.0001
            
        x_train, f_train, u250_train, u500_train, u750_train, u1000_train = data.minibatch()
        train_dict={f_ph: f_train, u250_ph: u250_train, u500_ph: u500_train, u750_ph: u750_train, u1000_ph: u1000_train, x_ph: x_train, learning_rate: lr}
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)

        if n%100 == 0:
            x_test, f_test, u250_test, u500_test, u750_test, u1000_test = data.testbatch(bs_test)
            u250_test_, u500_test_, u750_test_, u1000_test_ = sess.run([u250_pred, u500_pred, u750_pred, u1000_pred], feed_dict={f_ph: f_test, x_ph: x_test})
            # print(u250_test_.shape)
            # print(u250_test.shape)
            # sys.exit()
            # u_test = data.decoder(u_test)
            # u_test_ = data.decoder(u_test_)
            err250 = np.mean(np.linalg.norm(u250_test_ - u250_test, 2, axis=1)/np.linalg.norm(u250_test, 2, axis=1))
            err500 = np.mean(np.linalg.norm(u500_test_ - u500_test, 2, axis=1)/np.linalg.norm(u500_test, 2, axis=1))
            err750 = np.mean(np.linalg.norm(u750_test_ - u750_test, 2, axis=1)/np.linalg.norm(u750_test, 2, axis=1))
            err1000 = np.mean(np.linalg.norm(u1000_test_ - u1000_test, 2, axis=1)/np.linalg.norm(u1000_test, 2, axis=1))
            
            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, Loss: %.3e, Test L2 error 250: %.3f, Test L2 error 500: %.3f, Test L2 error 750: %.3f, Test L2 error 1000: %.3f, Time (secs): %.3f'%(n, loss_, err250, err500, err750, err1000, T))
            time_step_0 = time.perf_counter()
            test_loss[nt,0] = err250
            test_loss[nt,1] = err500
            test_loss[nt,2] = err750
            test_loss[nt,3] = err1000
            nt += 1
        
        train_loss[n,0] = loss_        
        n += 1
    
    save_models_to = save_results_to +"model/"
    if not os.path.exists(save_models_to):
        os.makedirs(save_models_to)      
    
    saver.save(sess, save_models_to+'Model')

    stop_time = time.perf_counter()
    print('Elapsed time (secs): %.3f'%(stop_time - start_time))
        
    np.savetxt(save_results_to+'/train_loss.txt', train_loss)
    np.savetxt(save_results_to+'/test_loss.txt', test_loss)

    data_save = SaveData()
    data_save.save(sess, x_ph, fnn_model, W_T, b_T, W_B, b_B, f_ph, u250_ph, u500_ph, u750_ph, u1000_ph, data, num_test, num_noisy, save_results_to)
    
    ## Plotting the loss history
    num_epoch = train_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, train_loss[:,0], color='blue', label='Training Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_his.png')

    ## Plotting the test loss
    num_epoch = test_loss.shape[0]
    x = np.linspace(1, num_epoch, num_epoch)
    fig = plt.figure(constrained_layout=False, figsize=(6, 6))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0])
    ax.plot(x, test_loss[:,0], color='blue', label='Testing Loss')
    ax.set_yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    ax.legend(loc='upper left')
    fig.savefig(save_results_to+'loss_test.png')    

if __name__ == "__main__":
    main()
