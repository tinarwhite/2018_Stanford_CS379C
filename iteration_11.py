import tensorflow as tf
import numpy as np
from movie_writer import visualize
from util import *

n_clusters = 2
n_hidden = 4
n_b_hidden = 4
saved_weights_init = np.load('converged_weights_11.npy') #switch

def single_cluster_network(input, n_hidden, trainflag):
    weights_layer1 = tf.get_variable("weights_layer1",(n_hidden,3),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    biases_layer1 = tf.get_variable("biases_layer1", (n_hidden, 1),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    weights_layer2 = tf.get_variable("weights_layer2",(n_hidden, n_hidden),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    biases_layer2 = tf.get_variable("biases_layer2", (n_hidden, 1),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    weights_layer3 = tf.get_variable("weights_layer3",(1, n_hidden),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    biases_layer3 = tf.get_variable("biases_layer3", (1, 1),trainable=trainflag,initializer=tf.random_uniform_initializer(-1,1))
    hidden_nodes_layer1 = tf.nn.tanh(tf.matmul(weights_layer1,input) + biases_layer1)
    hidden_nodes_layer2 = tf.nn.tanh(tf.matmul(weights_layer2,hidden_nodes_layer1) + biases_layer2)
    function_layer = tf.matmul(weights_layer3,hidden_nodes_layer2) + biases_layer3
    return function_layer

def network_architecture(n_clusters,n_hidden,n_b_hidden,input):
    functions = ["function1","function2"]
    clusters = ["cluster1","cluster2"]
    function_layers = [0,0]
    cluster_layers = [0,0]
    #functions = ["function1","function2","function3","function4"]
    #clusters = ["cluster1","cluster2","cluster3","cluster4"]
    #function_layers = [0,0,0,0]
    #cluster_layers = [0,0,0,0]
    for c, function in enumerate(functions):
        with tf.variable_scope(function):
            function_layers[c] = single_cluster_network(input, n_hidden, trainflag=True) #switch
    for c, cluster in enumerate(clusters):
        with tf.variable_scope(cluster):
            cluster_layers[c] = single_cluster_network(input, n_b_hidden, trainflag=True) #switch
    c = tf.nn.softmax(tf.concat(cluster_layers,0),0)
    f = tf.concat(function_layers,0)
    y = tf.reduce_sum(tf.multiply(c, f),0,keep_dims=True)
    return c, f, y

tf.reset_default_graph()
input = tf.placeholder("float32", [3, None])
# Build network
c, f, y = network_architecture(n_clusters,n_hidden,n_b_hidden,input)
# Objective functions definition
y_ = tf.placeholder("float32", [1, None])
# Choose best model with minimum error
#regression_loss = tf.reduce_min(tf.abs(y_-f),0) #switch
regression_loss = tf.abs(y_-y) #tf.square(y_-y) #switch
# Add up errors from all models to get loss
loss = tf.reduce_sum(regression_loss)
total_loss = loss #+ 0.1*tf.reduce_sum(regularization_loss)
# Training step type
opt1 = 0.001
opt2 = 0.8
opt3 = 0.2
train_step = tf.train.RMSPropOptimizer(opt1,opt2,opt3).minimize(total_loss) 

# Initialize network
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
n_trainable_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
[sess.run(var.assign(saved_weights_init[i])) for i, var in enumerate(tf.global_variables())] #switch, if save files exist

# Perceive and visualize directly
label = 'Burgers'
state = perceive(label, stride = 4)
stimuli, temporal_points, spatial_points, clips = state
#[visualize(state, label, stimulus) for stimulus in stimuli]

# Select stimuli to use for learning
chosen_stimuli = 0
stimuli_range = 3

# Select nonzero data to use for heuristic-based learning
batch_ratio = 0.1

# Send selected data through learning network
num_epochs = 500
total_error = np.zeros((num_epochs))
saved_weights = np.zeros((num_epochs,n_trainable_variables))
for j in range(num_epochs):
    if j % int(num_epochs/100) == 0:
        batch_inputs, batch_ys = train_batch(chosen_stimuli, stimuli_range, batch_ratio, state)
        theusual = {input: batch_inputs, y_:batch_ys}
    if j % 10 == 0:
        print("loss for epoch", j, "out of", num_epochs, sess.run(loss, feed_dict=theusual)/(np.prod(batch_ys.shape)))
    total_error[j] = sess.run(loss, feed_dict=theusual)/np.prod(batch_ys.shape)
    saved_weights[j] = np.hstack([sess.run(v).flatten() for v in tf.trainable_variables()])
    sess.run(train_step, feed_dict=theusual)

# Save weights only if they are well-converged
#np.save('converged_weights_11.npy',sess.run([var for var in tf.global_variables()]))

# Select all stimuli and representative time point for test final plotting
batch_inputs, batch_ys = train_batch(0, stimuli.shape[0], 1.0, state)
test_dict = {input: batch_inputs, y_:batch_ys}

# Select time point for representative plotting and plot type
time_plot = 50 #65 or 4 for either model
pflag = None

#Plot after only training function network
#predictions1 = sess.run(f, feed_dict=test_dict)[0].reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0])
#predictions2 = sess.run(f, feed_dict=test_dict)[1].reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0])
#plot_it(np.vstack((predictions1[:,time_plot], predictions2[:,time_plot], clips[:,time_plot,:])).T,'training_data',pflag)
#[plot_it(np.vstack((predictions1[list(stimuli).index(stimulus),::10], predictions2[list(stimuli).index(stimulus),::10],  clips[list(stimuli).index(stimulus),::10,:])).T,'training_data_%s' % stimulus,pflag) for stimulus in stimuli]

# Plot after training both networks
predictions = sess.run(y, feed_dict=test_dict).reshape(stimuli.shape[0],temporal_points.shape[0],spatial_points.shape[0])
learned_state = stimuli, temporal_points, spatial_points, predictions
plot_it(np.vstack((predictions[:,time_plot], clips[:,time_plot,:])).T,'training_data',pflag)
[plot_it(np.vstack((predictions[list(stimuli).index(stimulus),::10],  clips[list(stimuli).index(stimulus),::10,:])).T,'training_data_%s' % stimulus,pflag) for stimulus in stimuli]

# Plot for videos if desired
#[visualize(learned_state, 'Burgers_Learned', stimulus) for stimulus in stimuli]

## Plot only training stimuli for training plotting
#batch_inputs, batch_ys = train_batch(0, stimuli_range, 1.0, state)
#test_dict = {input: batch_inputs, y_:batch_ys}
#predictions = sess.run(y, feed_dict=test_dict).reshape(stimuli_range,temporal_points.shape[0],spatial_points.shape[0])
#[plot_it(np.vstack((predictions[list(stimuli).index(stimulus),::10], clips[list(stimuli).index(stimulus),::10,:])).T,'training_data_%s' % stimulus,pflag) for stimulus in stimuli[chosen_stimuli:chosen_stimuli+stimuli_range]]

# Plot error and weights
plot_it(total_error[9:],0)
plot_it(saved_weights,0)
