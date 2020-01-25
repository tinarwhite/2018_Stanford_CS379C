import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
import time

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def read_me(*args):
    return np.hstack([np.loadtxt(x)[:,20:200] for x in args])

def plot_it(v, i, flag = None, v_indices = np.array([0]), title = None, pltaxis = None):
    fig = plt.figure(figsize=(6,6)) #12,6
    plt.title(title,size=14)
    if np.sum(v_indices) == 0:
         vspace = np.linspace(0.0, 100.0, 1000)[:, None]
         plt.plot(vspace, v, lw=2)
         axes = plt.gca()
         #axes.set_xlim([xmin,xmax])
         axes.set_ylim([-1.5,1.5])
    else:
         vspace = v_indices
         plt.plot(vspace, v, lw=2)
    if pltaxis != None:
         plt.axis(pltaxis)
    if flag == 'save':
        my_path = 'C:\\Users\\Tina\\Desktop\\FRG\\NNs\\Project\\figs\\'
        plt.savefig(my_path + datetime.now().strftime('%H.%M') + '_' + str(i) + '_test.png')  
    else:
        plt.show()

def construct_fake_data(angle = 0, slope = 0):
    # Construct fake input data
    n_mu = 3
    n_t = 5
    n_x = 1000
    t = np.linspace(1.0, 5.0, n_t).reshape(n_t,1)
    x = np.linspace(0.0, 100.0, n_x).reshape(n_x,1)
    mu = np.linspace(1.0, 5.0, n_mu).reshape(n_mu,1)
    # Construct fake function data
    ind_center = 500
    ind_step0 = 100
    ind_step1 = 200
    ind_step05 = int((ind_step0+ind_step1)/2)
    yinit0 = 10.0
    yinit1 = 6.0
    yinit05 = (yinit0+yinit1)/2
    yend0 = 2.0
    yend1 = 4.0
    yend05 = (yend0+yend1)/2
    snaps = np.ones((n_x,n_t*n_mu)) 
    snaps[:ind_center-ind_step0*2,0] = yinit0+x[:ind_center-ind_step0*2,0]*slope
    snaps[:ind_center-ind_step0*1,1] = yinit0+x[:ind_center-ind_step0*1,0]*slope
    snaps[:ind_center-ind_step0*0,2] = yinit0+x[:ind_center-ind_step0*0,0]*slope
    snaps[:ind_center-ind_step0*-1,3] = yinit0+x[:ind_center-ind_step0*-1,0]*slope
    snaps[:ind_center-ind_step0*-2,4] = yinit0+x[:ind_center-ind_step0*-2,0]*slope
    snaps[:ind_center-ind_step05*2,5] = yinit05+x[:ind_center-ind_step05*2,0]*slope
    snaps[:ind_center-ind_step05*1,6] = yinit05+x[:ind_center-ind_step05*1,0]*slope
    snaps[:ind_center-ind_step05*0,7] = yinit05+x[:ind_center-ind_step05*0,0]*slope
    snaps[:ind_center-ind_step05*-1,8] = yinit05+x[:ind_center-ind_step05*-1,0]*slope
    snaps[:ind_center-ind_step05*-2,9] = yinit05+x[:ind_center-ind_step05*-2,0]*slope
    snaps[:ind_center-ind_step1*2,10] = yinit1+x[:ind_center-ind_step1*2,0]*slope
    snaps[:ind_center-ind_step1*1,11] = yinit1+x[:ind_center-ind_step1*1,0]*slope
    snaps[:ind_center-ind_step1*0,12] = yinit1+x[:ind_center-ind_step1*0,0]*slope
    snaps[:ind_center-ind_step1*-1,13] = yinit1+x[:ind_center-ind_step1*-1,0]*slope
    snaps[:ind_center-ind_step1*-2,14] = yinit1+x[:ind_center-ind_step1*-2,0]*slope
    snaps[ind_center-ind_step0*2:,0] = yend0+x[ind_center-ind_step0*2:,0]*slope
    snaps[ind_center-ind_step0*1:,1] = yend0+x[ind_center-ind_step0*1:,0]*slope
    snaps[ind_center-ind_step0*0:,2] = yend0+x[ind_center-ind_step0*0:,0]*slope
    snaps[ind_center-ind_step0*-1:,3] = yend0+x[ind_center-ind_step0*-1:,0]*slope
    snaps[ind_center-ind_step0*-2:,4] = yend0+x[ind_center-ind_step0*-2:,0]*slope
    snaps[ind_center-ind_step05*2:,5] = yend05+x[ind_center-ind_step05*2:,0]*slope
    snaps[ind_center-ind_step05*1:,6] = yend05+x[ind_center-ind_step05*1:,0]*slope
    snaps[ind_center-ind_step05*0:,7] = yend05+x[ind_center-ind_step05*0:,0]*slope
    snaps[ind_center-ind_step05*-1:,8] = yend05+x[ind_center-ind_step05*-1:,0]*slope
    snaps[ind_center-ind_step05*-2:,9] = yend05+x[ind_center-ind_step05*-2:,0]*slope
    snaps[ind_center-ind_step1*2:,10] = yend1+x[ind_center-ind_step1*2:,0]*slope
    snaps[ind_center-ind_step1*1:,11] = yend1+x[ind_center-ind_step1*1:,0]*slope
    snaps[ind_center-ind_step1*0:,12] = yend1+x[ind_center-ind_step1*0:,0]*slope
    snaps[ind_center-ind_step1*-1:,13] = yend1+x[ind_center-ind_step1*-1:,0]*slope
    snaps[ind_center-ind_step1*-2:,14] = yend1+x[ind_center-ind_step1*-2:,0]*slope
    # Reshape for neural network input
    input = x[::1].T
    output = snaps[::1].T
    # Scale to [0,1]
    scaled_input_1 = np.divide((input-input.min()), (input.max()-input.min()))
    scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
    # Scale to [-1,1]
    scaled_input_2 = (scaled_input_1*2)-1
    scaled_output_2 = (scaled_output_1*2)-1
    # Decide which scale to use
    input_data = scaled_input_2
    output_data = scaled_output_2
    # Rotate for 3 cluster test case
    input1 = np.cos(angle*np.pi/180)*input_data - np.sin(angle*np.pi/180)*output_data
    output1 = np.sin(angle*np.pi/180)*input_data + np.cos(angle*np.pi/180)*output_data
    # Project back to the same x components
    output2 = np.zeros_like(output)
    for i in range(snaps.shape[1]):
        output2[i] = np.interp(input_data,input1[i],output1[i])

    input2 = input_data
    # Train on only a single state
    input_data_train = input_data
    #plot_it(output2.T,0)
    return input2, output2, t, mu

def construct_fake_data_old(slope = 0):
    # Construct fake input data
    n_mu = 3
    n_t = 5
    n_x = 1000
    t = np.linspace(1.0, 5.0, n_t).reshape(n_t,1)
    x = np.linspace(0.0, 100.0, n_x).reshape(n_x,1)
    mu = np.linspace(1.0, 5.0, n_mu).reshape(n_mu,1)
    # Construct fake function data
    ind_center = 500
    ind_step0 = 100
    ind_step1 = 200
    ind_step05 = int((ind_step0+ind_step1)/2)
    yinit0 = 10.0
    yinit1 = 6.0
    yinit05 = (yinit0+yinit1)/2
    yend0 = 2.0
    yend1 = 4.0
    yend05 = (yend0+yend1)/2
    snaps = np.ones((n_x,n_t*n_mu)) 
    snaps[:ind_center-ind_step0*2,0] = yinit0+x[:ind_center-ind_step0*2,0]*slope
    snaps[:ind_center-ind_step0*1,1] = yinit0+x[:ind_center-ind_step0*1,0]*slope
    snaps[:ind_center-ind_step0*0,2] = yinit0+x[:ind_center-ind_step0*0,0]*slope
    snaps[:ind_center-ind_step0*-1,3] = yinit0+x[:ind_center-ind_step0*-1,0]*slope
    snaps[:ind_center-ind_step0*-2,4] = yinit0+x[:ind_center-ind_step0*-2,0]*slope
    snaps[:ind_center-ind_step05*2,5] = yinit05+x[:ind_center-ind_step05*2,0]*slope
    snaps[:ind_center-ind_step05*1,6] = yinit05+x[:ind_center-ind_step05*1,0]*slope
    snaps[:ind_center-ind_step05*0,7] = yinit05+x[:ind_center-ind_step05*0,0]*slope
    snaps[:ind_center-ind_step05*-1,8] = yinit05+x[:ind_center-ind_step05*-1,0]*slope
    snaps[:ind_center-ind_step05*-2,9] = yinit05+x[:ind_center-ind_step05*-2,0]*slope
    snaps[:ind_center-ind_step1*2,10] = yinit1+x[:ind_center-ind_step1*2,0]*slope
    snaps[:ind_center-ind_step1*1,11] = yinit1+x[:ind_center-ind_step1*1,0]*slope
    snaps[:ind_center-ind_step1*0,12] = yinit1+x[:ind_center-ind_step1*0,0]*slope
    snaps[:ind_center-ind_step1*-1,13] = yinit1+x[:ind_center-ind_step1*-1,0]*slope
    snaps[:ind_center-ind_step1*-2,14] = yinit1+x[:ind_center-ind_step1*-2,0]*slope
    snaps[ind_center-ind_step0*2:,0] = yend0+x[ind_center-ind_step0*2:,0]*slope
    snaps[ind_center-ind_step0*1:,1] = yend0+x[ind_center-ind_step0*1:,0]*slope
    snaps[ind_center-ind_step0*0:,2] = yend0+x[ind_center-ind_step0*0:,0]*slope
    snaps[ind_center-ind_step0*-1:,3] = yend0+x[ind_center-ind_step0*-1:,0]*slope
    snaps[ind_center-ind_step0*-2:,4] = yend0+x[ind_center-ind_step0*-2:,0]*slope
    snaps[ind_center-ind_step05*2:,5] = yend05+x[ind_center-ind_step05*2:,0]*slope
    snaps[ind_center-ind_step05*1:,6] = yend05+x[ind_center-ind_step05*1:,0]*slope
    snaps[ind_center-ind_step05*0:,7] = yend05+x[ind_center-ind_step05*0:,0]*slope
    snaps[ind_center-ind_step05*-1:,8] = yend05+x[ind_center-ind_step05*-1:,0]*slope
    snaps[ind_center-ind_step05*-2:,9] = yend05+x[ind_center-ind_step05*-2:,0]*slope
    snaps[ind_center-ind_step1*2:,10] = yend1+x[ind_center-ind_step1*2:,0]*slope
    snaps[ind_center-ind_step1*1:,11] = yend1+x[ind_center-ind_step1*1:,0]*slope
    snaps[ind_center-ind_step1*0:,12] = yend1+x[ind_center-ind_step1*0:,0]*slope
    snaps[ind_center-ind_step1*-1:,13] = yend1+x[ind_center-ind_step1*-1:,0]*slope
    snaps[ind_center-ind_step1*-2:,14] = yend1+x[ind_center-ind_step1*-2:,0]*slope
    # Reshape for neural network input
    output = snaps[::1].T
    input = x[::1].T
    # Rotate for 3 cluster test case
    np.sin(angle*np.pi/180)
    # Scale to [0,1]
    scaled_input_1 = np.divide((input-input.min()), (input.max()-input.min()))
    scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
    # Scale to [-1,1]
    scaled_input_2 = (scaled_input_1*2)-1
    scaled_output_2 = (scaled_output_1*2)-1
    # Decide which scale to use
    input_data = scaled_input_2
    output_data = scaled_output_2
    # Train on only a single state
    input_data_train = input_data
    return input_data_train, output_data, t, mu

def construct_fake_data_oldest():
    # Construct fake input data
    n_mu = 3
    n_t = 5
    n_x = 1000
    t = np.linspace(1.0, 5.0, n_t).reshape(n_t,1)
    x = np.linspace(0.0, 100.0, n_x).reshape(n_x,1)
    mu = np.linspace(1.0, 5.0, n_mu).reshape(n_mu,1)
    # Construct fake function data
    ind_center = 500
    ind_step0 = 100
    ind_step1 = 200
    ind_step05 = int((ind_step0+ind_step1)/2)
    yinit0 = 10.0
    yinit1 = 6.0
    yinit05 = (yinit0+yinit1)/2
    yend0 = 2.0
    yend1 = 4.0
    yend05 = (yend0+yend1)/2
    snaps = np.ones((n_x,n_t*n_mu)) 
    snaps[:ind_center-ind_step0*2,0] = yinit0
    snaps[:ind_center-ind_step0*1,1] = yinit0
    snaps[:ind_center-ind_step0*0,2] = yinit0
    snaps[:ind_center-ind_step0*-1,3] = yinit0
    snaps[:ind_center-ind_step0*-2,4] = yinit0
    snaps[:ind_center-ind_step05*2,5] = yinit05
    snaps[:ind_center-ind_step05*1,6] = yinit05
    snaps[:ind_center-ind_step05*0,7] = yinit05
    snaps[:ind_center-ind_step05*-1,8] = yinit05
    snaps[:ind_center-ind_step05*-2,9] = yinit05
    snaps[:ind_center-ind_step1*2,10] = yinit1
    snaps[:ind_center-ind_step1*1,11] = yinit1
    snaps[:ind_center-ind_step1*0,12] = yinit1
    snaps[:ind_center-ind_step1*-1,13] = yinit1
    snaps[:ind_center-ind_step1*-2,14] = yinit1
    snaps[ind_center-ind_step0*2:,0] = yend0
    snaps[ind_center-ind_step0*1:,1] = yend0
    snaps[ind_center-ind_step0*0:,2] = yend0
    snaps[ind_center-ind_step0*-1:,3] = yend0
    snaps[ind_center-ind_step0*-2:,4] = yend0
    snaps[ind_center-ind_step05*2:,5] = yend05
    snaps[ind_center-ind_step05*1:,6] = yend05
    snaps[ind_center-ind_step05*0:,7] = yend05
    snaps[ind_center-ind_step05*-1:,8] = yend05
    snaps[ind_center-ind_step05*-2:,9] = yend05
    snaps[ind_center-ind_step1*2:,10] = yend1
    snaps[ind_center-ind_step1*1:,11] = yend1
    snaps[ind_center-ind_step1*0:,12] = yend1
    snaps[ind_center-ind_step1*-1:,13] = yend1
    snaps[ind_center-ind_step1*-2:,14] = yend1
    # Reshape for neural network input
    output = snaps[::1].T
    input = x[::1].T
    # Scale to [0,1]
    scaled_input_1 = np.divide((input-input.min()), (input.max()-input.min()))
    scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
    # Scale to [-1,1]
    scaled_input_2 = (scaled_input_1*2)-1
    scaled_output_2 = (scaled_output_1*2)-1
    # Decide which scale to use
    input_data = scaled_input_2
    output_data = scaled_output_2
    # Train on only a single state
    input_data_train = input_data
    return input_data_train, output_data, t, mu
    
def import_burgers_data(stride=1):
    # Code here for importing data from file
    snaps =  read_me('burg/snaps_0p02_0p02_5.dat',
                     'burg/snaps_0p02_0p02_1.dat',
                     'burg/snaps_0p02_0p02_2p5.dat').T

    mu = np.array((5,1,2.5))
    (n_samp, n_x), n_mu = snaps.shape, 3
    n_t = int(n_samp/n_mu)

    t = np.linspace(0.0, 500.0, n_t)
    x = np.linspace(0.0, 100.0, n_x)

    # Make inputs noisy
    noisy_input = x
    output = snaps
    # Scale to [0,1]
    scaled_t_1 = np.divide((t-t.min()), (t.max()-t.min()))
    scaled_input_1 = np.divide((noisy_input-noisy_input.min()), (noisy_input.max()-noisy_input.min()))
    scaled_output_1 = np.divide((output-output.min()), (output.max()-output.min()))
    # Scale to [-1,1]
    scaled_t_2 = (scaled_t_1*2)-1
    scaled_input_2 = (scaled_input_1*2)-1
    scaled_output_2 = (scaled_output_1*2)-1

    # What to output
    input_t = scaled_t_2[::stride]
    #input_t = t.reshape(1,t.shape[0])
    input_data_train = scaled_input_2.reshape(1,n_x)
    output_data = scaled_output_2[::stride]

    #plot_it(output_data.T,0)
    return input_data_train, output_data, input_t, mu
