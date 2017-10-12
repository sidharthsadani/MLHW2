'''
*****************************************************************
AUTHOR : Sidharth Sadani (1503352), Reza (PS #)
DATE : 10/12/17
FILE : Q1a.py
COMMENTS : Part 1a, Assignment 2, COSC6342
******************************************************************
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os
# Defining Summaries Directory to store tensorboard data
summaries_dir = os.path.join(os.getcwd(), 'summaries_dir/Q1a')

# Load Data Set
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
# Define Tensorflow Session
sess = tf.Session()

print("Tensorflowa & MNIST Data Imported! Hello World!")

# Defining Input Vector
x = tf.placeholder(tf.float32, shape=[None, 784])
# Images of size 28x28 are linearized to 784
# Defining The Target Vector : Ground Truth Outputs
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Defining the model parameters
# Weights : Initialized to 0
W = tf.Variable(tf.zeros([784,10]))
# Adding variables for tensorflow visualization
#tf.summary.histogram('Weights', W)


# Biases : Initialized to 0
b = tf.Variable(tf.zeros([10]))
#tf.summary.histogram('Biases', b)

print("Placeholders & Variables Defined")

# Before variables can be used in a session, they must be initialized using that session
sess.run(tf.global_variables_initializer())

# Define The Regression Model
y = tf.matmul(x, W) + b

# Defining The Loss Function
# Here we use cross entropy function between the target and output
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		labels=y_, logits=y))
# Tensorboard Variable for Cross Entropy
tf.summary.scalar('Loss : Cross Entropy', cross_entropy)

print("Loss Function Defined")

# Defining The Training Step of the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Tensorboard Variable for Accuracy
tf.summary.scalar('Accuracy', accuracy)


## Merging All The Summaries and write them out to /tmp/mnnist_logs
merged = tf.summary.merge_all()
# Tensorboard Writer : Training Measures
train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
# Tensorboard Writer : Testing Measures
test_writer = tf.summary.FileWriter(summaries_dir + '/test')

# Training the model can be done by repeatedly running the train_step
with sess.as_default():
	# Initialize all the variables in the graph
	tf.global_variables_initializer().run()
	for _ in range(1000):
		batch = mnist.train.next_batch(100)
		# Performing Training Step & Measuring accuracy
		summary, acc = sess.run([merged, train_step],
			feed_dict={x: batch[0], y_: batch[1]})
		# Tensorboard output for training at each iteration
		train_writer.add_summary(summary, _)
		# train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		
		# Perform Testing after every 10 steps
		if(_ %10 == 0):
			summary, acc = sess.run([merged, accuracy], feed_dict={
				x: mnist.test.images, y_: mnist.test.labels})
			# Tensorboard output for testing at every 10 training iterations
			test_writer.add_summary(summary, _)
		
	print("Training Done")
	
	print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print("All Done!!")
