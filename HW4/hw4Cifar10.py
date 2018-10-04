# David Levi
# Project 3

# Used a youtube tutorial to handle the Cifar loading as well as a few of the hyperparamers (https://www.youtube.com/watch?v=7TlkKd4vf4o).
# I also combined this with my project from last week, yielding around 65% accuracy.  Obviously not great, but
# it was definitely harder to do some trial and error due to very long training times which, due to the jewish holidays
# I didnt really have time for.

import os
import pickle
import tensorflow as tf
import numpy as np 
from tqdm import tqdm

from keras.utils import to_categorical


class CifarLoader(object):
	def __init__(self, sourceFile):
		self.source = sourceFile
		self.i = 0
		self.images = None
		self.labels = None

	def load(self):
		data = [unpickle(f) for f in self.source]
		images = np.vstack([d[b"data"] for d in data])
		n = len(images)
		self.images = images.reshape(n,3,32,32).transpose(0,2,3,1).astype(float)/255
		self.labels = one_hot(np.hstack([d[b"labels"] for d in data]), 10)
		return self

	def get_batch(self, batch_size):
		x, y = self.images[self.i:self.i+batch_size], self.labels[self.i:self.i+batch_size]
		self.i = (self.i+batch_size)%len(self.images)
		return x,y

class CifarData(object):
	def __init__(self):
		self.train = CifarLoader(["data_batch_{}".format(i) for i in range(1,6)]).load()
		self.test = CifarLoader(["test_batch"]).load()



PATH = "cifar10"
NUM_CLASSES = 10

def unpickle(file):
	with open(os.path.join(PATH,file), 'rb') as fo:
		dict = pickle.load(fo, encoding="bytes")
	return dict

def one_hot(vec, vals=NUM_CLASSES):
	n = len(vec)
	out = np.zeros((n,vals))
	out[range(n), vec] = 1
	return out
# END OF TUTORIAL HELP

cifar = CifarData()

NUM_CLASSES = 10
BATCH_SIZE = 128
KEEP_RATE = 0.5
NUM_EPOCHS = 10

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)




def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn(x):
	w_dict = {
			'w_conv1':tf.Variable(tf.truncated_normal([5,5,3,32],stddev=0.1)),
			'w_conv2':tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1)),
			'w_connected':tf.Variable(tf.truncated_normal([8*8*64,1024],stddev=0.1)), # 28/4
			'w_output':tf.Variable(tf.truncated_normal([1024,NUM_CLASSES],stddev=0.1))
	}

	norm = 0
	#for w in w_dict:
		#norm+=tf.pow(tf.norm(w_dict[w]),2)

	b_dict = {
			'b_conv1':tf.Variable(tf.constant(0.1,shape=[32])),
			'b_conv2':tf.Variable(tf.constant(0.1,shape=[64])),
			'b_connected':tf.Variable(tf.constant(0.1,shape=[1024])),
			'b_output':tf.Variable(tf.constant(0.1,shape=[NUM_CLASSES])),

	}

	x = tf.reshape(x,shape=[-1,32,32,3])

	# 1st Convolutional Layer
	C1 = tf.nn.relu(conv2d(x,w_dict['w_conv1'])+b_dict['b_conv1'])
	C1 = maxpool2d(C1)

	# 2nd Convolutional Layer
	C2 = tf.nn.relu(conv2d(C1,w_dict['w_conv2'])+b_dict['b_conv2'])
	C2 = maxpool2d(C2)

	# Connected Layer
	LC = tf.reshape(C2,[-1, 8*8*64])
	LC = tf.nn.relu(tf.matmul(LC,w_dict['w_connected'])+b_dict['b_connected'])
	LC = tf.nn.dropout(LC, KEEP_RATE)

	out = tf.matmul(LC, w_dict['w_output'])+b_dict['b_output']

	return out,norm

def train(x):
	pred,norm = cnn(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(NUM_EPOCHS):
			currLoss = 0
			currBatch = 0
			valLoss = 0

			
			for _ in tqdm(range(int(len(cifar.train.images)/BATCH_SIZE)-20)):
				#epoch_x = x_train[currBatch:currBatch+BATCH_SIZE]
				#epoch_y = y_train[currBatch:currBatch+BATCH_SIZE]
				epoch_x,epoch_y = cifar.train.get_batch(BATCH_SIZE)
				currBatch+=BATCH_SIZE
				_, lossTrain = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				currLoss += lossTrain

			currBatch = 0

			for _ in range(20):
				epoch_x,epoch_y = cifar.train.get_batch(BATCH_SIZE)
				
				currBatch+=BATCH_SIZE
				lossVal = sess.run(cost, feed_dict={x: epoch_x, y: epoch_y})
				valLoss += lossVal


			print('Epoch', epoch, 'completed out of',NUM_EPOCHS,'loss:',currLoss,'val_Loss:',valLoss)

		x_test,y_test = cifar.test.get_batch(10000)

		correct = tf.nn.in_top_k(predictions=pred,targets=tf.argmax(y,1),k=1)
		correct5 = tf.nn.in_top_k(predictions=pred,targets=tf.argmax(y,1),k=5)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		accuracy5 = tf.reduce_mean(tf.cast(correct5, 'float'))
		#accuracy = correct

		#accuracy = tf.metrics.mean(tf.nn.in_top_k(predictions=pred, targets=yN, k=5))
		print('Top Accuracy: ',accuracy.eval({x:x_test, y:y_test}))
		print('Top 5 Accuracy:',accuracy5.eval({x:x_test, y:y_test}))

train(x)






































