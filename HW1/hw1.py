import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
import operator

from tqdm import tqdm

NUM_FEATURES = 6
BATCH_SIZE = 32
NUM_BATCHES = 50000

# Pure sin wave
class CleanData(object):
	def __init__(self,sigma=0.1):
		num_samp = 50

		self.index = np.arange(0,1,.01)

		self.x = self.index
		self.y = np.sin(2*np.pi*self.x)

	def get_batch(self):
		
		return self.x,self.y
		
# Noisy data
class Data(object):
	def __init__(self,sigma=0.1):
		num_samp = 50
		sigma = sigma
		np.random.seed(31415)

		e = np.random.uniform(0,sigma,num_samp)

		self.index = np.arange(num_samp)

		self.x = np.random.uniform(0,1,num_samp)
		self.y = np.sin(2*np.pi*self.x)+e

	def get_batch(self):
		inds = np.arange(BATCH_SIZE)
		np.random.shuffle(inds)
		return self.x[inds], self.y[inds]

# Gaussians
def f(x):
	b = tf.get_variable('b',[],tf.float32,tf.zeros_initializer())
	mu = tf.get_variable('mu',[NUM_FEATURES,1],tf.float32,tf.random_normal_initializer())
	sig = tf.get_variable('sig',[NUM_FEATURES,1],tf.float32,tf.random_normal_initializer())
	w = tf.get_variable('w',[1,NUM_FEATURES],tf.float32,tf.random_normal_initializer())

	return tf.matmul(w,(tf.exp(-1*((x-mu)**2)/sig**2)))+b
	
	
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
y_hat = f(x)

loss = tf.reduce_mean(tf.pow(y_hat-y,2)/2)
optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Plotting clean data
dataClean = CleanData(0)
xClean,yClean = dataClean.get_batch();

# Plotting noisy data
data = Data()
xG,yG = data.get_batch()

for _ in tqdm(range(0, NUM_BATCHES)):
	x_np, y_np = data.get_batch()
	loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})
	#print(loss_np)


print("Parameter estimates:")
params = {}
for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
	params[var.name.rstrip(":0")] = np.array(sess.run(var)).flatten()
	print(var.name.rstrip(":0"),
		np.array_str(np.array(sess.run(var)).flatten(),precision=3))

# Plotting basis vectors
xNew = np.arange(0,1,.01)
basis_sum = 0
features = ['w','b','sig','mu']
base = []
for featNum in range(NUM_FEATURES):
	base_sum = params['w'][featNum] * np.exp(-1*((xNew - params['mu'][featNum])**2)/(params['sig'][featNum]**2))
	basis_sum+= base_sum
	base.append(base_sum)
yNew = basis_sum+params['b'][0]

# Actual plotting (Figure 1)
plt.figure()
plt.plot(xClean,yClean,'b-')
plt.plot(xG,yG,'go')
plt.plot(xNew,yNew,'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fit')
plt.savefig("figure1.png", format="png")
plt.show()

# Actual plotting (Figure 2)
plt.figure()
for b in base:
	plt.plot(xNew,b)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bases for fit')
plt.savefig("figure2.png", format = "png")
plt.show()










