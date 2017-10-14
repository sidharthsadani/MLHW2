'''
*****************************************************************
AUTHOR : Sidharth Sadani (1503352), Reza (PS #)
DATE : 10/12/17
FILE : Q1b.py
COMMENTS : Part 1b, Assignment 2, COSC6342
******************************************************************
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
# Defining Summaries Directory to store tensorboard data
summaries_dir = os.path.join(os.getcwd(), 'summaries_dir/Q1b')

### Function Definitions
# Functions for weights & Bias Initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Defining Convolution & Pooling Layers
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Load Data Set
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

batch = mnist.train.next_batch(3)

print("Testing")
# Define Tensorflow Session
sess = tf.Session()

# Tensorflow Placeholders for Variables
# Input Placeholder
x = tf.placeholder(tf.float32, shape = [None, 784]) 
# Output Placeholder
y_ = tf.placeholder(tf.float32, shape = [None, 10])


# Defining The First Convolution Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshaping The Image to a 4d Tensor
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolving x_image with the weight tensor & applying ReLU
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# Linearizing the pooling layer 2
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# Final One-Hot Output
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


print("Hello World!")


# Loss Function Definition
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	labels=y_, logits=y_conv))
# Since Testing is done batchwise, we create a placeholder for the tensorboard variable
cross_entropy_value = tf.placeholder(tf.float32, shape=())
cross_entropy_summary = tf.summary.scalar('Loss : Cross Entropy', cross_entropy_value)
# Optimizer Used : Adaptive Momentum
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Since Testing is done batchwise, we create a placeholder for the tensorboard variable
accuracy_value = tf.placeholder(tf.float32, shape=())
accuracy_summary = tf.summary.scalar('Accuracy', accuracy_value)

# Merging All The Summaries
merged = tf.summary.merge_all()
# Tensorboard Writer : Training Measures
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
# Tensorboard Writer : Testing Measures
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

print("Setup Complete")

# Begin Training & Testing

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		#### TRAINING CODE BEGIN #####
		batch = mnist.train.next_batch(50)
		# Capture Accuracy and Entropy on Training Data Every 10 steps.
		if i % 10 == 0:
			# Compute Accuracy on current batch using network so far 
			train_accuracy, train_cross_ent = sess.run([accuracy, cross_entropy], feed_dict={
				x: batch[0], y_: batch[1], keep_prob: 1.0})
			# Compute Tensorboard Summary and perform training step for current batch
			summary, _ = sess.run([merged, train_step],
				feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5,
					accuracy_value: train_accuracy, cross_entropy_value: train_cross_ent})
			# Write Tensorboard Summary Data
			train_writer.add_summary(summary, i)
		else:
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		#### TRAINING CODE END ######
		#### TESTING CODE BEGIN #####
		# Capture Accuracy and Entropy on Testing Data Every 100 Steps
		if(i % 100 == 0):
			## Since on our implementation the GPU was running out of memory we do the testing in batches as well
			print('step %d, training accuracy %g' % (i, train_accuracy))
			counter = 0 # Keep track of loop iterations, for batchwise testing
			total_samples = 0  # Keep track of number of test images seen
			cross_ent_sum = 0 # Sum of cross entropy so far.
			# Since batch sizes are equal mean of entropy across batches equals the mean entropy
			# Obtained using all test samples at once
			sum_correct = 0 # No of correct classifications in this batch
			test_size = 10000 # Total Test Images Available
			batch_size = 50 # Size of each test batch
			while True:
				if(counter == test_size/batch_size):
					break
				counter += 1
				if(counter % 100 == 0):
					# Just to test if loop is working correctly
					print("Loop Counter: %d" % counter)
				try:
					# Extract Next batch of images
					testSet = mnist.test.next_batch(batch_size)
				except:
					break
				# Compute Accuracy & Entropy for the current batch
				iter_accuracy, iter_cross_ent = sess.run([accuracy, cross_entropy],
					feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0})
				# Variables to keep track of overall accuracy and entropy
				total_samples += batch_size
				sum_correct += batch_size*float(iter_accuracy)
				cross_ent_sum += iter_cross_ent
			test_accuracy = sum_correct/total_samples
			test_cross_ent = cross_ent_sum/counter
			#print('Iter Test accuracy %g' % iter_accuracy)
			print("Total Test Accuracy: ", test_accuracy)
			# Save this test run results on the tensorboard
			summary = sess.run(merged, feed_dict={
				x: testSet[0], y_: testSet[1], keep_prob: 1.0, 
				accuracy_value: test_accuracy, cross_entropy_value: test_cross_ent})
			test_writer.add_summary(summary, i)
		#### TESTING CODE END ####
	print("Training Complete")
