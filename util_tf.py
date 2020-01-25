import tensorflow as tf
import numpy as np
from util import *

def single_cluster_network(input, n_hidden, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(n_hidden,3),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer1 = tf.get_variable("biases_layer1", (n_hidden, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    weights_layer2 = tf.get_variable("weights_layer2",(n_hidden, n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer2 = tf.get_variable("biases_layer2", (n_hidden, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    weights_layer3 = tf.get_variable("weights_layer3",(1, n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer3 = tf.get_variable("biases_layer3", (1, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    hidden_nodes_layer1 = tf.nn.tanh(tf.matmul(weights_layer1,input) + biases_layer1)
    hidden_nodes_layer2 = tf.nn.tanh(tf.matmul(weights_layer2,hidden_nodes_layer1) + biases_layer2)
    function_layer = tf.matmul(weights_layer3,hidden_nodes_layer2) + biases_layer3
    return function_layer

def single_linear_cluster_network(input, n_hidden, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(1,3),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer1 = tf.get_variable("biases_layer1", (1, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    function_layer = tf.matmul(weights_layer1,input) + biases_layer1
    return function_layer

def single_3D_cluster_network(input, n_hidden, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(n_hidden,5),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer1 = tf.get_variable("biases_layer1", (n_hidden, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    weights_layer2 = tf.get_variable("weights_layer2",(n_hidden, n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer2 = tf.get_variable("biases_layer2", (n_hidden, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    weights_layer3 = tf.get_variable("weights_layer3",(1, n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer3 = tf.get_variable("biases_layer3", (1, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    hidden_nodes_layer1 = tf.nn.tanh(tf.matmul(weights_layer1,input) + biases_layer1)
    hidden_nodes_layer2 = tf.nn.tanh(tf.matmul(weights_layer2,hidden_nodes_layer1) + biases_layer2)
    function_layer = tf.matmul(weights_layer3,hidden_nodes_layer2) + biases_layer3
    return function_layer

def single_3D_linear_cluster_network(input, n_hidden, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(1,5),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer1 = tf.get_variable("biases_layer1", (1, 1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    function_layer = tf.matmul(weights_layer1,input) + biases_layer1
    return function_layer

def network_3D_architecture(n_clusters,n_hidden,n_b_hidden,input):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    clusters = ["cluster%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    cluster_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_3D_cluster_network(input, n_hidden, trainflag=True) #switch
    for c, cluster in enumerate(clusters):
        with tf.variable_scope(cluster):
            cluster_layers[c] = single_3D_cluster_network(input, n_b_hidden, trainflag=True) #switch
    c = tf.nn.softmax(tf.concat(cluster_layers,0),0)
    #c = tf.concat(cluster_layers,0)
    f = tf.concat(function_layers,0)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,0,keep_dims=True)
    return c, f, y, fc

def network_architecture(n_clusters,n_hidden,n_b_hidden,input):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    clusters = ["cluster%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    cluster_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_cluster_network(input, n_hidden, trainflag=True) #switch
    for c, cluster in enumerate(clusters):
        with tf.variable_scope(cluster):
            cluster_layers[c] = single_cluster_network(input, n_b_hidden, trainflag=True) #switch
    c = tf.nn.softmax(tf.concat(cluster_layers,0),0)
    #c = tf.concat(cluster_layers,0)
    f = tf.concat(function_layers,0)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,0,keep_dims=True)
    return c, f, y, fc

def linear_3D_network_architecture(n_clusters,n_hidden,n_b_hidden,input):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    clusters = ["cluster%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    cluster_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_3D_linear_cluster_network(input, n_hidden, trainflag=True) #switch
    for c, cluster in enumerate(clusters):
        with tf.variable_scope(cluster):
            cluster_layers[c] = single_3D_cluster_network(input, n_b_hidden, trainflag=True) #switch
    c = tf.nn.softmax(tf.concat(cluster_layers,0),0)
    #c = tf.concat(cluster_layers,0)
    f = tf.concat(function_layers,0)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,0,keep_dims=True)
    return c, f, y, fc

def linear_network_architecture(n_clusters,n_hidden,n_b_hidden,input):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    clusters = ["cluster%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    cluster_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_linear_cluster_network(input, n_hidden, trainflag=True) #switch
    for c, cluster in enumerate(clusters):
        with tf.variable_scope(cluster):
            cluster_layers[c] = single_cluster_network(input, n_b_hidden, trainflag=True) #switch
    c = tf.nn.softmax(tf.concat(cluster_layers,0),0)
    #c = tf.concat(cluster_layers,0)
    f = tf.concat(function_layers,0)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,0,keep_dims=True)
    return c, f, y, fc

def RNN(input, n_hidden, n_output, timesteps, weights, biases, lstm_scope):
    # input is a list of 'timesteps' tensors of shape (batch_size, n_input)
    #unstacked_input = tf.unstack(input, timesteps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    # Get lstm cell output
    print(input)
    #print(unstacked_input[0])
    #static_outputs, static_states = tf.contrib.rnn.static_rnn(lstm_cell, unstacked_input, dtype=tf.float32)
    with tf.variable_scope(lstm_scope):
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    reshaped_weights = tf.tile(tf.reshape(weights,[1,n_output,n_hidden]),[timesteps,1,1])
    reshaped_outputs = tf.transpose(outputs,[1,2,0])
    hidden_lstm_layer = tf.transpose(tf.matmul(reshaped_weights,reshaped_outputs),[2,0,1])+biases
    return hidden_lstm_layer

def lstm_single_cluster_network(input, n_hidden, timesteps, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(n_hidden,n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer1 = tf.get_variable("biases_layer1", (n_hidden),initializer=tf.random_uniform_initializer(-1.0,1.0))
    function_layer1 = RNN(input, n_hidden, n_hidden, timesteps, weights_layer1, biases_layer1,'lstm1')
    weights_layer2 = tf.get_variable("weights_layer2",(n_hidden,1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    biases_layer2 = tf.get_variable("biases_layer2", (1),initializer=tf.random_uniform_initializer(-1.0,1.0))
    function_layer2 = RNN(function_layer1, n_hidden, 1, timesteps, weights_layer2, biases_layer2,'lstm2')
    return function_layer2

def lstm_network_architecture(n_clusters,n_hidden,n_b_hidden,input,timesteps):
    functions = ["function%s"% str(n) for n in range(n_clusters)]
    clusters = ["cluster%s"% str(n) for n in range(n_clusters)]
    function_layers = [0] * n_clusters
    cluster_layers = [0] * n_clusters
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = lstm_single_cluster_network(input, n_hidden, timesteps, trainflag=True) #switch
    f = tf.concat(function_layers,2)
    if n_clusters > 1: 
        for c, cluster in enumerate(clusters):
            with tf.variable_scope(cluster):
                cluster_layers[c] = lstm_single_cluster_network(input, n_b_hidden, timesteps, trainflag=True) #switch
        c = tf.nn.softmax(tf.concat(cluster_layers,2),2)
    if n_clusters == 1: 
        c = tf.ones_like(f)
    fc = tf.multiply(c, f)
    y = tf.reduce_sum(fc,2,keep_dims=True)
    return c, f, y, fc

def forcing_function_network(input_masked, n_functions, limits):
    mu, t, x = tf.split(input_masked,[1,1,1])
    vars = tf.get_variable("gaussian_vars",(4,n_functions),initializer=tf.random_uniform_initializer(-2.0,2.0))
    center0, speed, width, height = tf.split(tfsigmoid_limit(vars,limits[:,:1],limits[:,1:]),[1,1,1,1])
    g = tfgaussian_wave(x, t, center0, speed, width, height)
    return g

def forcing_function_sin_network(input_masked, n_functions, limits):
    mu, t, x = tf.split(input_masked,[1,1,1])
    vars = tf.get_variable("sin_vars",(4,n_functions),initializer=tf.random_uniform_initializer(-2.0,2.0))
    center0, speed, width, height = tf.split(tfsigmoid_limit(vars,limits[:,:1],limits[:,1:]),[1,1,1,1])
    g = tfsin_wave(x, t, center0, speed, width, height)
    return g

def tfsigmoid_limit(vars,min,max):
    limit_vars = tf.multiply(tf.sigmoid(vars),max-min)+min
    return limit_vars

def tfsin_wave(x, t, center0, speed, width, height):
    center = tf.transpose(center0) + tf.transpose(speed)*5*t
    g = tf.transpose(height)*tf.sin((x-center)/tf.transpose(width))
    return g

def tfgaussian_wave(x, t, center0, speed, width, height):
    center = tf.transpose(center0) + tf.transpose(speed)*t
    g = tf.transpose(height)*tf.exp(-0.5*((x-center)/(tf.transpose(width)/6))**2)
    return g

def descriptive_finance_clusters(n_clusters,stimuli_range,state,f_run,y__run):
    argmin2 = np.argmin(np.abs(y__run-f_run),0).reshape(stimuli_range,state[3].shape[1]*state[3].shape[2])
    mu_exp = [[np.float(np.sum(argmin2[i] == j)) for i in range(stimuli_range)] for j in range(n_clusters)] # to determine explanatory power in each mu
    heuristic = [(np.min(mu_exp[j])*np.sum(mu_exp[j]))/(np.max(mu_exp[j])*np.sum(mu_exp)) for j in range(n_clusters)]
    return heuristic, [i for i, x in enumerate(np.array(heuristic) > 0.05/n_clusters) if x]

def determine_n_clusters(sess,n_clusters,stimuli_range,state,f_run,y__run):
    heuristic, kept_clusters = descriptive_finance_clusters(n_clusters,stimuli_range,state,f_run,y__run)
    save_vars = sum([tf.get_collection('trainable_variables',"function%s"% str(n)) for n in kept_clusters],[])\
              + sum([tf.get_collection('trainable_variables',"cluster%s"% str(n)) for n in kept_clusters],[])
    n_clusters_network = len(kept_clusters)
    print('Training found %s clusters'%n_clusters_network)
    if n_clusters_network != n_clusters:
        print("Reducing number of clusters from %s to %s" % (n_clusters,n_clusters_network))
    return heuristic, n_clusters_network, save_vars

def descriptive_clusters(n_clusters,stimuli_range,state,f_run,y__run):
    argmin2 = np.argmin(np.abs(y__run-f_run),0).reshape(stimuli_range,state[3].shape[1]*state[3].shape[2])
    mu_exp = [[np.float(np.sum(argmin2[i] == j)) for i in range(stimuli_range)] for j in range(n_clusters)] # to determine explanatory power in each mu
    heuristic = [(np.min(mu_exp[j])*np.sum(mu_exp[j]))/(np.max(mu_exp[j])*np.sum(mu_exp)) for j in range(n_clusters)]
    return heuristic, [i for i, x in enumerate(np.array(heuristic) > 0.05/n_clusters) if x]

def lstm_determine_n_clusters(sess,n_clusters,stimuli_range,state,f_run,y__run):
    heuristic, kept_clusters = descriptive_clusters(n_clusters,stimuli_range,state,f_run,y__run)
    save_vars = sum([tf.get_collection('trainable_variables',"function%s"% str(n)) for n in kept_clusters],[])\
              + sum([tf.get_collection('trainable_variables',"cluster%s"% str(n)) for n in kept_clusters],[])
    n_clusters_network = len(kept_clusters)
    print('Training found %s clusters'%n_clusters_network)
    if n_clusters_network != n_clusters:
        print("Reducing number of clusters from %s to %s" % (n_clusters,n_clusters_network))
    return heuristic, n_clusters_network, save_vars

def scramble_masked_input_network(input_x,input_mu_ts,y_,spatial_points,mask_x,trainflag=True):
    scramble_weights = tf.get_variable("scramble_weights",(1,len(spatial_points)),trainable=trainflag,initializer=tf.random_uniform_initializer(0.999,1.001))
    scramble_biases = tf.get_variable("scramble_biases", (1,len(spatial_points)),trainable=trainflag,initializer=tf.random_uniform_initializer(-0.001,0.001))
    n_repeat = tf.cast(tf.shape(input_mu_ts)[1]/tf.shape(input_x)[1],"int32")
    mask_y = tf.reshape(tf.tile(mask_x,[n_repeat]),(1,len(spatial_points)*n_repeat))
    input_xp = tf.sin(6.28318*(tf.multiply(scramble_weights,input_x)+scramble_biases))
    unscaled = tf.multiply(scramble_weights,input_x)+scramble_biases
    input_xp = unscaled
    #mean, var = tf.nn.moments(unscaled,1)
    #input_xp = (unscaled-mean)*tf.sqrt(0.25)/tf.sqrt(var)
    #input_xp = tf.divide((unscaled-tf.reduce_min(unscaled)), (tf.reduce_max(unscaled)-tf.reduce_min(unscaled)))*2-1
    input_xp_tile = tf.tile(input_xp,[1,n_repeat])
    input = tf.concat([input_mu_ts,input_xp_tile],0)
    mask_input = tf.concat([mask_y,mask_y,mask_y],0)
    input_masked_unknown = tf.boolean_mask(input, mask_input)
    n_mask = tf.cast(tf.shape(input_masked_unknown)[0]/3,"int32")
    input_masked = tf.reshape(input_masked_unknown,(3,n_mask))
    y__masked = tf.boolean_mask(y_, mask_y)
    return input_xp, input_masked, y__masked, n_repeat, n_mask

def scramble_masked_3D_input_network(input_x,input_mu_ts,y_,spatial_points,mask_x,trainflag=True):
    scramble_weights = tf.get_variable("scramble_weights",(3,len(spatial_points)),trainable=trainflag,initializer=tf.random_uniform_initializer(0.999,1.001))
    scramble_biases = tf.get_variable("scramble_biases", (3,len(spatial_points)),trainable=trainflag,initializer=tf.random_uniform_initializer(-0.001,0.001))
    n_repeat = tf.cast(tf.shape(input_mu_ts)[1]/tf.shape(input_x)[1],"int32")
    mask_y = tf.reshape(tf.tile(mask_x,[n_repeat]),(1,len(spatial_points)*n_repeat))
    #input_xp = tf.sin(6.28318*(tf.multiply(scramble_weights,input_x)+scramble_biases))
    unscaled = tf.multiply(scramble_weights,input_x)+scramble_biases
    input_xp = unscaled
    #mean, var = tf.nn.moments(unscaled,1)
    #input_xp = (unscaled-mean)*tf.sqrt(0.25)/tf.sqrt(var)
    #input_xp = tf.divide((unscaled-tf.reduce_min(unscaled)), (tf.reduce_max(unscaled)-tf.reduce_min(unscaled)))*2-1
    input_xp_tile = tf.tile(input_xp,[1,n_repeat])
    input = tf.concat([input_mu_ts,input_xp_tile],0)
    mask_input = tf.concat([mask_y,mask_y,mask_y,mask_y,mask_y],0)
    input_masked_unknown = tf.boolean_mask(input, mask_input)
    n_mask = tf.cast(tf.shape(input_masked_unknown)[0]/5,"int32")
    input_masked = tf.reshape(input_masked_unknown,(5,n_mask))
    y__masked = tf.boolean_mask(y_, mask_y)
    return input_xp, input_masked, y__masked, n_repeat, n_mask

def get_center_diffs(input_masked,y__masked,f,n_clusters, n_repeat, n_mask):
    _,_,x_masked = tf.split(input_masked,[1,1,1])
    argmin_y = tf.argmin(tf.abs(y__masked-f),0)
    argmin_bool = tf.convert_to_tensor([tf.cast(tf.equal(argmin_y,i), tf.float32) for i in np.arange(n_clusters)])
    x_bool = tf.convert_to_tensor([tf.cast(tf.equal(argmin_y,i), tf.float32)*tf.squeeze(x_masked,0) for i in np.arange(n_clusters)])
    centers_calc = tf.reshape(tf.reduce_mean(x_bool,1),(n_clusters,1))
    mean, var = tf.nn.moments(centers_calc,0)
    centers = (centers_calc-mean)*tf.sqrt(0.25)/tf.sqrt(var)
    #centers = tf.reshape(tf.convert_to_tensor(np.linspace(-1,1,n_clusters*2+1,dtype=np.float32)[1::2]),(n_clusters,1))
    center_diffs = tf.multiply(x_masked-centers_calc,argmin_bool) #centers_calc works well, but change to centers
    ## move x only by the center it most often appears in across time
    #input_shape = tf.cast(tf.shape(y__masked)[0]/n_repeat,"int32")
    #center_inds = tf.argmax(tf.reduce_sum(tf.reshape(argmin_bool,(n_clusters,input_shape,n_repeat)),2),0)
    #center_inds_tiled = tf.tile(center_inds,[n_repeat])
    #centers_new = tf.reshape(tf.gather(tf.reshape(centers,(n_clusters,)),center_inds_tiled),(1,n_mask))
    #center_diffs = x_masked-centers_new
    return center_diffs

#[9936] vs. [8,16560]