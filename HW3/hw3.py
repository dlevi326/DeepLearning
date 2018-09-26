# David Levi
# Project 3

import tensorflow as tf
import numpy as np 
from tqdm import tqdm

from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

val_split = int(len(x_train)*.8)

x_val = x_train[val_split:]
y_val = y_train[val_split:]

x_train = x_train[:val_split]
y_train = y_train[:val_split]

y_val = to_categorical(y_val)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

NUM_CLASSES = 10
BATCH_SIZE = 128
KEEP_RATE = 0.8
NUM_EPOCHS = 10

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)




def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn(x):
	w_dict = {
			'w_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			'w_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			'w_connected':tf.Variable(tf.random_normal([7*7*64,1024])), # 28/4
			'w_output':tf.Variable(tf.random_normal([1024,NUM_CLASSES]))
	}

	norm = 0
	for w in w_dict:
		norm+=tf.pow(tf.norm(w_dict[w]),2)

	b_dict = {
			'b_conv1':tf.Variable(tf.random_normal([32])),
			'b_conv2':tf.Variable(tf.random_normal([64])),
			'b_connected':tf.Variable(tf.random_normal([1024])),
			'b_output':tf.Variable(tf.random_normal([NUM_CLASSES])),

	}

	x = tf.reshape(x,shape=[-1,28,28,1])

	# 1st Convolutional Layer
	C1 = tf.nn.relu(conv2d(x,w_dict['w_conv1'])+b_dict['b_conv1'])
	C1 = maxpool2d(C1)

	# 2nd Convolutional Layer
	C2 = tf.nn.relu(conv2d(C1,w_dict['w_conv2'])+b_dict['b_conv2'])
	C2 = maxpool2d(C2)

	# Connected Layer
	LC = tf.reshape(C2,[-1, 7*7*64])
	LC = tf.nn.relu(tf.matmul(LC,w_dict['w_connected'])+b_dict['b_connected'])
	LC = tf.nn.dropout(LC, KEEP_RATE)

	out = tf.matmul(LC, w_dict['w_output'])+b_dict['b_output']

	return out,norm

def train(x):
	pred,norm = cnn(x)
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=pred))
	optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(NUM_EPOCHS):
			currLoss = 0
			currBatch = 0
			valLoss = 0

			
			for _ in tqdm(range(int(len(x_train)/BATCH_SIZE))):
				epoch_x = x_train[currBatch:currBatch+BATCH_SIZE]
				epoch_y = y_train[currBatch:currBatch+BATCH_SIZE]
				currBatch+=BATCH_SIZE
				_, lossTrain = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				currLoss += lossTrain

			currBatch = 0

			for _ in range(int(len(x_val)/BATCH_SIZE)):
				epoch_x = x_val[currBatch:currBatch+BATCH_SIZE]
				epoch_y = y_val[currBatch:currBatch+BATCH_SIZE]
				currBatch+=BATCH_SIZE
				lossVal = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y})
				valLoss += lossVal


			print('Epoch', epoch, 'completed out of',NUM_EPOCHS,'loss:',currLoss,'val_Loss:',valLoss)

		correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

train(x)




























