
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import rnnutils as ru

#TODO: Suppress Tensorflow warnings

import os
import random
print("Hello World, RNN!")

dH = ru.RNNUtils(inpFile="FrostInput.txt")
dH.parseData()

print("Data Parsing Done!")

# Defining Netowrk Parameters
num_input = dH.vocabSize # No of Inputs / Size of Embedding
num_classes = dH.vocabSize # Size of output / Size of Embedding
num_layers = 2 # No of Hidden Layers, Since we require Multi-Layer NN
num_hidden = 256 # Number of Hidden Neurons (LSTM Cells)
num_unroll = 100 # No of Times the RNN is unrolled
dH.setNumUnroll(num_unroll)
batch_size = 64 # No of times the unrolled RNN is replicated for training
dH.setBatchSize(batch_size)

# Learning Rate
lr = 0.001
# Loop Control To Control Number of Epochs, for simplicity we count the number of batches
# As opposed to epochs
num_batches = 100


####### BEGIN RNN Class Definition ########
class MyRNN:
    def __init__(self, input_size, output_size, num_layers, num_hidden, session, lr):
        
        self.num_input = input_size
        self.num_classes = output_size
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.session = session
        self.learning_rate = lr
        self.name_scope = "MyRNN"
        self.prev_state = np.zeros((self.num_layers*self.num_hidden*2,))
        # print(self.prev_state.shape)

        with tf.variable_scope(self.name_scope):
            # Placeholder for input
            self.x = tf.placeholder(tf.float32, shape=(None, None, self.num_input))
            # Placeholder for state
            self.init_state = tf.placeholder(tf.float32, shape=(None, self.num_layers*self.num_hidden*2))
            # Placeholder for Output
            self.y_ = tf.placeholder(tf.float32, shape=(None, None, self.num_classes))
            # LSTM Has 2 state variables(c,h) for each hidden neuron, hence placeholder size = 2*(layers)*(neurons_per_layer)
            # Usually it is stored as a tuple, but we unpack it and thus set the state_tupe = False while defining the cells
            # Defining LSTM Cells/Layers
            self.cells = [rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=False) for i in range(self.num_layers)]

            # Defining LSTM Network
            # Wrapping the two cells into a multicell RNN LSTM
            self.lstm = rnn.MultiRNNCell(self.cells, state_is_tuple=False)

            self.outputs, self.next_state = tf.nn.dynamic_rnn(self.lstm, self.x, 
                    initial_state = self.init_state, dtype=tf.float32)

            self.Wy = tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
            self.By = tf.Variable(tf.random_normal([self.num_classes]))
            
            # logits = tf.matmul(outputs[-1], self.Wy) + self.By
            # Setting Network for batch processing
            # To store outputs of every single LSTM, across unrolling and batches
            self.outputs_batch = tf.reshape(self.outputs, [-1, self.num_hidden])
            batch_logits = tf.matmul(self.outputs_batch, self.Wy) + self.By

            oSh = tf.shape(self.outputs)
            all_pred = tf.nn.softmax(batch_logits)
            # Softmax Predictions of every single LSTM across unrolling and batches
            self.final_pred = tf.reshape(all_pred, [oSh[0], oSh[1], self.num_classes])

            # Reshaping Output Placeholder for batch training
            # This way we feed only one linearized input and the network will reshape it appropriately in batches
            y_batch = tf.reshape(self.y_, [-1, self.num_classes])

            # Defining Loss Function
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=batch_logits, labels=y_batch))
            # TODO: A
            # Define Training Step and Optimizer
            self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cross_entropy)
    
    def stepPred(self, Inp, firstIter):
        if firstIter:
            cur_state = np.zeros((self.num_layers*self.num_hidden*2,))
        else:
            cur_state = self.prev_state

        # Using the previous state & Inpute Generate Next State and Output
        out, next_state = self.session.run([self.final_pred, self.next_state], feed_dict={
            self.x: x, self.init_state:cur_state})
            
        self.next_state = next_state[0]

        return out[0][0]

    def stepTrain(self, xb, yb):
        batchSize = xb.shape[0]
        init_state = np.zeros(shape=(batchSize, self.num_layers*self.num_hidden*2), 
                dtype=np.float32)

        ce_loss, _ = self.session.run([self.cross_entropy, self.train_step], feed_dict={
            self.x:xb, self.y_: yb, self.init_state: init_state})

        return ce_loss




####### END RNN Class Definition ##########

# sess = tf.Session()

with tf.Session() as sess:
    net = MyRNN(num_input, num_classes, num_layers, num_hidden, sess, lr)
    sess.run(tf.global_variables_initializer())
    # Define Network Saver
    saver = tf.train.Saver(tf.global_variables())
    for i in range(num_batches):
        # get Batch Seed Points
        x,y = dH.nextBatchNew()
        # print(x.shape)
        # print(y.shape)
        # print(i)
        # Train Network
        cross_ent = net.stepTrain(x,y)
        if(i%10==0):
            print("Iteration: %d, Entropy_Loss: %.4f" % (i, cross_ent))
        # Save Network
