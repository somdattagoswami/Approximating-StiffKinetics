import tensorflow.compat.v1 as tf
import numpy as np
import scipy
import sys
from fnn import FNN

class SaveData:
    def __init__(self):
        pass

    def save(self, sess, x_ph, fnn_model, W_T, b_T, W_B, b_B, f_ph, u250_ph, u500_ph, u750_ph, u1000_ph, data, num_test, num_noisy, save_results_to):
        
        num_test = 5000*12
        x_num = 12
        x_test, f_test, u250_test, u500_test, u750_test, u1000_test = data.testbatch_print(num_test)
        
        test_dict = {f_ph: f_test, x_ph: x_test}
        u_T = fnn_model.fnn(W_T, b_T, x_ph) 
        u_B = fnn_model.fnn_B(W_B, b_B, f_ph)   
    
        u_pred = tf.einsum('ijk,mk->ikm',u_B,u_T)
        
        u250_pred = u_pred[:,0:12,:]  
        u500_pred = u_pred[:,12:24,:]
        u750_pred = u_pred[:,24:36,:]
        u1000_pred = u_pred[:,36:48,:]
    
        u250_pred = tf.reduce_sum(u250_pred, axis=-1, keepdims=True)
        u500_pred = tf.reduce_sum(u500_pred, axis=-1, keepdims=True)
        u750_pred = tf.reduce_sum(u750_pred, axis=-1, keepdims=True)
        u1000_pred = tf.reduce_sum(u1000_pred, axis=-1, keepdims=True)
       
        u250_pred_, u500_pred_, u750_pred_, u1000_pred_ = sess.run([u250_pred, u500_pred, u750_pred, u1000_pred], feed_dict={f_ph: f_test, x_ph: x_test})
        
        u250_test, u500_test, u750_test, u1000_test = data.decoder(u250_test, u500_test, u750_test, u1000_test)        
        u250_pred_, u500_pred_, u750_pred_, u1000_pred_  = data.decoder(u250_pred_, u500_pred_, u750_pred_, u1000_pred_)

        u250_pred_ = np.reshape(u250_pred_, (num_test, x_num))
        u500_pred_ = np.reshape(u500_pred_, (num_test, x_num))
        u750_pred_ = np.reshape(u750_pred_, (num_test, x_num))
        u1000_pred_ = np.reshape(u1000_pred_, (num_test, x_num))

        U250_ref = np.reshape(u250_test, (num_test, x_num))
        U500_ref = np.reshape(u500_test, (num_test, x_num))
        U750_ref = np.reshape(u750_test, (num_test, x_num))
        U1000_ref = np.reshape(u1000_test, (num_test, x_num))
        
        f_test = np.reshape(f_test, (f_test.shape[0], -1))

        err1 = np.mean(np.linalg.norm(u250_pred_ - U250_ref, 2, axis=1)/np.linalg.norm(U250_ref, 2, axis=1))
        err2 = np.mean(np.linalg.norm(u500_pred_ - U500_ref, 2, axis=1)/np.linalg.norm(U500_ref, 2, axis=1))
        err3 = np.mean(np.linalg.norm(u750_pred_ - U750_ref, 2, axis=1)/np.linalg.norm(U750_ref, 2, axis=1))
        err4 = np.mean(np.linalg.norm(u1000_pred_ - U1000_ref, 2, axis=1)/np.linalg.norm(U1000_ref, 2, axis=1))
        
        
        print('Relative L2 Error T250: %.3f'%(err1))
        print('Relative L2 Error T500: %.3f'%(err2))
        print('Relative L2 Error T750: %.3f'%(err3))
        print('Relative L2 Error T1000: %.3f'%(err4))
        
        err1 = np.reshape(err1, (-1, 1))
        err2 = np.reshape(err2, (-1, 1))
        err3 = np.reshape(err3, (-1, 1))
        err4 = np.reshape(err4, (-1, 1))

        
        np.savetxt(save_results_to+'/err1', err1, fmt='%e')
        np.savetxt(save_results_to+'/err2', err2, fmt='%e')
        np.savetxt(save_results_to+'/err3', err3, fmt='%e')
        np.savetxt(save_results_to+'/err4', err4, fmt='%e')

        
        save_dict = {'u250_pred': u250_pred_, 'u250_ref': U250_ref,\
                     'u500_pred': u500_pred_, 'u500_ref': U500_ref,\
                     'u750_pred': u750_pred_, 'u750_ref': U750_ref,\
                     'u1000_pred': u1000_pred_, 'u1000_ref': U1000_ref, 'f_test': f_test}
        
        scipy.io.savemat(save_results_to+'/pred.mat', save_dict)


