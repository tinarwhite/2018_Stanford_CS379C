import os
import tensorflow as tf
import numpy as np
from scipy.stats import linregress
from util import *
from util_tf import *
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

starttime = time()

# Perceive and visualize directly
label = 'Burgers'
state = perceive(label, stride = 4)
stimuli, temporal_points, spatial_points, clips = state
#[visualize(state, label, stimulus) for stimulus in stimuli]
timesteps = temporal_points.shape[0]

# Learning control variables
learn_until_converged = True
learn_cluster_or_function = 'initialize'
num_epochs_print = 5000
chosen_stimuli, stimuli_range = (0,3)
n_hidden, n_b_hidden = (3,3)
n_clusters_network = 1
batch_ratio = 0.05

# Logic and system variables
savefilename = 'converged_weights_lstm_2.npy'
last_j = 0
num_epochs = 1000000 if learn_until_converged == True else num_epochs_print
total_error = np.zeros((num_epochs))
saved_weights = np.zeros((num_epochs,1000))

# Network construction and training loop
converged = False
while converged == False:
    # Build clustered network architecture
    print('building clustered network architecture...')
    tf.reset_default_graph()
    n_clusters = n_clusters_network
    input = tf.placeholder("float32", [None, timesteps, 2])
    y_ = tf.placeholder("float32", [None, timesteps, 1])
    c, f, y, fc = lstm_network_architecture(n_clusters,n_hidden,n_b_hidden,input,timesteps)
    # Organize variables into groups
    train_vars_f = tf.get_collection('trainable_variables',"function")
    train_vars_c = tf.get_collection('trainable_variables',"cluster")
    train_vars_all_weights = [v for v in tf.trainable_variables() if 'bias' not in v.name]
    train_vars_layer1_weights = [v for v in train_vars_all_weights if 'layer1' in v.name]
    # Choose a loss function
    regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in train_vars_all_weights]) * 0.001
    loss_min = tf.reduce_sum(tf.reduce_min(tf.abs(y_-f),2))#+regularization_loss
    loss = tf.reduce_sum(tf.abs(y_-y))#+regularization_loss
    # Training step definition
    learning_rate, decay, momentum = (0.001, 0.8, 0.7)
    train_step = tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(loss_min)
    if n_clusters != 1:
        train_step_c = tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(loss, var_list=train_vars_c)
    #train_step_f = tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(loss, var_list=train_vars_f)
    #train_step_all = tf.train.RMSPropOptimizer(learning_rate, decay, momentum).minimize(loss)
    # Initialize network
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # Set learning type and training parameters
    batch_inputs, batch_ys = lstm_train_batch(0, stimuli_range, 1.0, state)
    test_dict = {input: batch_inputs, y_:batch_ys}
    if learn_cluster_or_function == 'initialize': 
        loss_run, train_step_run = (loss_min, train_step)
    if learn_cluster_or_function == 'cluster': 
        loss_run, train_step_run = (loss, train_step_c)
    if learn_cluster_or_function == 'function': 
        loss_run, train_step_run = (loss, train_step_f)
    if learn_cluster_or_function == 'both': 
        loss_run, train_step_run = (loss, train_step_all)
    # Load memory if needed
    if os.path.isfile(savefilename) == True:
        saved_weights_init = np.load(savefilename)
        [sess.run(var.assign(saved_weights_init[i])) for i, var in enumerate(tf.trainable_variables())] 
    # Train clustered network
    num_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('training...')
    for j in range(last_j,num_epochs):
        if j % 10 == 0:
            batch_inputs, batch_ys = lstm_train_batch(chosen_stimuli, stimuli_range, batch_ratio, state)
            theusual = {input: batch_inputs, y_:batch_ys}
        if j % 10 == 0:
            if num_epochs == 1000000:
                print("Loss for epoch", j, "out of unknown # is", sess.run(loss_run, feed_dict=theusual)/(np.prod(batch_ys.shape)))
            else:
                print("Loss for epoch", j, "out of", num_epochs, "is", sess.run(loss_run, feed_dict=theusual)/(np.prod(batch_ys.shape)))            
        total_error[j] = sess.run(loss_run, feed_dict=theusual)/np.prod(batch_ys.shape)
        saved_weights[j,:num_variables] = np.hstack([sess.run(v).flatten() for v in tf.trainable_variables()])
        sess.run(train_step_run, feed_dict=theusual)
        if learn_until_converged == True:
            if j % 100 == 0 and j > 499 + last_j:
                slope = np.abs(linregress(range(500), total_error[j-500:j]).slope)
                slope_desired = 0.5*10**-7
                print('Convergence slope is %s out of %s'%(slope,slope_desired))
                if slope < slope_desired:
                    print('Converged!')
                    last_j = j
                    num_epochs_print = j
                    break
    # Determine number of clusters to keep and save weights for only those clusters
    f_run = np.transpose(sess.run(f,test_dict).reshape(stimuli_range,spatial_points.shape[0],temporal_points.shape[0],n_clusters),[3,0,2,1])
    y__run = np.transpose(sess.run(y_,test_dict).reshape(stimuli_range,spatial_points.shape[0],temporal_points.shape[0],1),[3,0,2,1])
    heuristic, n_clusters_network, save_vars = lstm_determine_n_clusters(sess,n_clusters,stimuli_range,state,f_run,y__run)
    np.save(savefilename,sess.run([var for var in tf.trainable_variables()]))
    # Logic for total convergence
    converged = True if learn_until_converged == False else False
    if learn_cluster_or_function == 'cluster':
        converged = True
    if n_clusters_network == n_clusters and learn_until_converged == True:
        learn_cluster_or_function = 'cluster'
    if n_clusters == 1:
        converged = True

# Calculate time taken
endtime = time()
seconds = int(endtime-starttime)
m, s = divmod(seconds, 60)
print('Training took %s minutes %s seconds'%(m,s))

# Select time point for representative plotting and plot type
time_plot = 50 #65 or 4 for either model
pflag = None

# Plot after only training function network
if learn_cluster_or_function == 'initialize':
    argmin = np.argmin(np.abs(y__run-f_run),0)
    unique, counts = np.unique(argmin, return_counts=True)
    reconstructions = f_run[0]
    for i in range(stimuli_range):
        for j in range(temporal_points.shape[0]):
            for k in range(spatial_points.shape[0]):
                reconstructions[i,j,k] = f_run[argmin[i,j,k],i,j,k]
    plot_it(np.vstack((reconstructions[:,time_plot,:], clips[:,time_plot,:])).T,spatial_points,'training_data',pflag)
    [plot_it(np.vstack((f_run[:,j,time_plot,:], clips[j,time_plot,:])).T,spatial_points,'training_data',pflag) for j in range(stimuli_range)]

# Plot after training both networks
if learn_cluster_or_function != 'initialize':
    batch_inputs, batch_ys = lstm_train_batch(0, stimuli.shape[0], 1.0, state)
    total_dict = {input: batch_inputs, y_:batch_ys}
    predictions = np.transpose(sess.run(y, feed_dict=total_dict).reshape(stimuli.shape[0],spatial_points.shape[0],temporal_points.shape[0]),[0,2,1])
    plot_it(np.vstack((predictions[:,time_plot], clips[:,time_plot,:])).T,spatial_points,0,'training_data',pflag)
    [plot_it(np.vstack((predictions[list(stimuli).index(stimulus),::10],  clips[list(stimuli).index(stimulus),::10,:])).T,spatial_points,0,'training_data_%s' % stimulus,pflag) for stimulus in stimuli]

# Plot error and weights
plot_it(total_error[9:num_epochs_print])
#plot_it(saved_weights[:num_epochs_print])
print('Training took %s minutes %s seconds and found %s clusters'%(m,s,n_clusters_network))

