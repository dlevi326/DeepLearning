# David Levi
# Project 2

import tensorflow as tf 

import numpy as np 
import matplotlib.pyplot as plt 
import operator

from tqdm import tqdm

NUM_FEATURES = 40
NUM_BATCHES = 40000

# Used your idea of creating two data classes
class Hyp0(object):
	def __init__(self,n_points,noise=.5):
		self.n_points = n_points
		self.noise = noise
    
	def get_batch(self):
		degrees = 360
		n = (np.sqrt(np.random.rand(self.n_points,1)) * 780 * (2*np.pi)/degrees)
		d1x = -np.cos(n)*n + np.random.rand(self.n_points,1) * self.noise
		d1y = np.sin(n)*n + np.random.rand(self.n_points,1) * self.noise
		return d1x,d1y


class Hyp1(object):
	def __init__(self,n_points,noise=.5):
		self.n_points = n_points
		self.noise = noise

	def get_batch(self):
		degrees = 360
		n = (np.sqrt(np.random.rand(self.n_points,1)) * 780 * (2*np.pi)/degrees)+1
		d1x = (-np.cos(n)*n + np.random.rand(self.n_points,1) * self.noise)
		d1y = (np.sin(n)*n + np.random.rand(self.n_points,1) * self.noise)
		return -d1x,-d1y


def f(x):
	'''
	Finding f(x) was definitely difficult.  I ended up making a simple 2 layer network with 40 features, this ended up
	being good enough for learning a few spirals
	'''

	#x2 = tf.sin(x)
	#x3 = tf.cos(x)

	with tf.variable_scope('f',reuse=tf.AUTO_REUSE):
		b1 = tf.get_variable('b1',[],tf.float32,tf.ones_initializer())
		b2 = tf.get_variable('b2',[],tf.float32,tf.ones_initializer())
		w1 = tf.get_variable('w1',[2,NUM_FEATURES],tf.float32,tf.random_normal_initializer())
		w2 = tf.get_variable('w2',[NUM_FEATURES,1],tf.float32,tf.random_normal_initializer())
	
	L2 = (tf.sigmoid(tf.matmul(x,w1)+b1))
	hy = (tf.matmul(L2,w2)+b2)

	return(hy)


H0 = Hyp0(1000)
H1 = Hyp1(1000)
x1, y1 = H0.get_batch()
spiral1 = np.hstack((x1,y1))
x2, y2 = H1.get_batch()
spiral2 = np.hstack((x2,y2))

plt.plot(x1,y1,'.')
plt.plot(x2,y2,'.')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spirals before fitting')
plt.savefig("Spirals.png", format = "png")
plt.show()


xFunc = tf.placeholder(tf.float32)
y_act = tf.placeholder(tf.float32)
hy = f(xFunc)


norm = 0
norm += tf.pow(tf.norm(tf.get_variable('w1',[2,NUM_FEATURES],tf.float32,tf.random_normal_initializer())),2)
norm += tf.pow(tf.norm(tf.get_variable('w2',[NUM_FEATURES,2],tf.float32,tf.random_normal_initializer())),2)


loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_act, logits=hy)) + (.01 * norm)


optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()



sess = tf.Session()
sess.run(init)


inds = np.random.randint(2,size=NUM_BATCHES)

for index,_ in enumerate(tqdm(range(0, NUM_BATCHES))):
	if inds[index] == 0:
		x_n, y_n = H0.get_batch()
		x_np = np.array(np.hstack((x_n,y_n)))
		loss_np, _ = sess.run([loss, optim], feed_dict={xFunc: x_np, y_act: 0})
	elif inds[index]==1:
		x_n, y_n = H1.get_batch()
		x_np = np.array(np.hstack((x_n,y_n)))
		loss_np, _ = sess.run([loss, optim], feed_dict={xFunc: x_np, y_act: 1})
	if index%10000==0:
		print(loss_np)
print(loss_np)


print("Parameter estimates:")
params = {}
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
	params[var.name.rstrip(":0")] = np.array(sess.run(var)).flatten()
	print(var.name.rstrip(":0"),
		np.array_str(np.array(sess.run(var)).flatten(),precision=3))



# Got help with creating the meshgrid
xGrid,yGrid = np.meshgrid(np.arange(-15,15,.01), np.arange(-15,15,.01))
inputx = np.stack((xGrid,yGrid), axis=2)
y_est = np.array([])
xOut = tf.placeholder(tf.float32)
y_hat = f(xOut)

for i in tqdm(range(inputx.shape[0])):
    if y_est.size>0: 
        y_est = np.hstack((y_est,sess.run(y_hat,{xOut: inputx[i]})))
    else: 
        y_est = sess.run(y_hat,{xOut:inputx[i]})

print(y_est)
y_est[y_est>0] = 1
y_est[y_est<=0] = 0
plt.contourf(yGrid,xGrid,y_est)
plt.plot(x1, y1, ".")
plt.plot(x2, y2, ".")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Spiral Fit')
plt.savefig("SpiralFit.png", format = "png")
plt.show()

