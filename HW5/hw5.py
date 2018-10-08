# David Levi
# Project 5
# AG News Classification

import os
import pickle
import tensorflow as tf
import numpy as np 
from tqdm import tqdm
from math import log,ceil
import pandas as pd
from nltk.corpus import stopwords
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import csv

import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

NUM_CLASSES = 4 # World Sports Business Sci/Tech
BATCH_SIZE = 15 # Num sentences to take a
KEEP_RATE = 0.5
NUM_EPOCHS = 1

class CsvLoader(object):
	def __init__(self,filename):
		self.file = open(filename, mode='r')
		self.batchNum = 0

	def printLines(self):
		csv_reader = csv.DictReader(self.file)

		line_count=0
		for row in csv_reader:
			if(line_count>10):
				break
			print('-'*60)
			print(row['Category'])
			print(row['Title'])
			print(row['Summary'])
			line_count += 1

	def getAllUniqueWords(self,lines):
		csv_reader = csv.DictReader(self.file,fieldnames=['Category','Title','Summary'])
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words) + list(set(manual_stops))
		#wordsTitle = []
		word_list = []
		words = []
		sentences = []

		curr_line = 0
		for row in csv_reader:
			if(curr_line>lines):
			    break
			sentTitle = nltk.word_tokenize(row['Title'])

			wordsTitle = []
			for w in sentTitle:
				if w not in ignore_words:
					wSplit = w.split('\\')
					for w1 in wSplit:
						wordsTitle.append(stemmer.stem(w1.lower()))

			sentSumm = nltk.word_tokenize(row['Summary'])
			wordsSumm = []
			for w in sentSumm:
				if w not in ignore_words:
					wSplit = w.split('\\')
					for w1 in wSplit:
						wordsSumm.append(stemmer.stem(w1.lower()))

			sentences.append(list(wordsTitle)+list(wordsSumm))

			#wordsTitle = [stemmer.stem(w1.lower()) for w1 in w.split('\\') for w in sentTitle if w not in ignore_words]
			#sentSumm = nltk.word_tokenize(row['Summary'])
			#wordsSumm = [stemmer.stem(w.lower()) for w in sentSumm if w not in ignore_words]
			word_list = list(set(wordsTitle))+list(set(wordsSumm))
			curr_line+=1
			words+=word_list
		    
		print (len(set(words)))
		return set(words),sentences

	def getData(self,embeddedDict):
		# Returns a list of ordered dicts

		csv_reader = csv.DictReader(self.file,fieldnames=['Category','Title','Summary'])
		totalArr = []
		ignore_words = set(stopwords.words('english'))
		manual_stops = ['(',')']
		ignore_words = list(ignore_words) + list(set(manual_stops))

		line_count = self.batchNum
		curr_line = 0
		for row in csv_reader:
			tempDict = []
			if(curr_line>line_count+BATCH_SIZE):
				print('breaking at line',curr_line,'where break is:',line_count+BATCH_SIZE)
				break

			#while(curr_line<line_count):
			#	curr_line+=1
			#	continue


			sentTitle = nltk.word_tokenize(row['Title'])

			wordsTitle = []
			for w in sentTitle:
				if w not in ignore_words:
					wSplit = w.split('\\')
					for w1 in wSplit:
						wordsTitle.append(stemmer.stem(w1.lower()))

			sentSumm = nltk.word_tokenize(row['Summary'])
			wordsSumm = []
			for w in sentSumm:
				if w not in ignore_words:
					wSplit = w.split('\\')
					for w1 in wSplit:
						wordsSumm.append(stemmer.stem(w1.lower()))

			tempDict.append(wordsTitle+wordsSumm)
			tempDict2 = []
			for w in tempDict[0]:
				if w in embeddedDict:
					tempDict2.append(embeddedDict[w])
				else:
					print(w)
					tempDict2.append(embeddedDict['UNKOWNWORD'])

			curr_line+=1
			totalArr.append(np.reshape(np.array(tempDict2),[-1,len(embeddedDict['UNKOWNWORD'])]))


		self.batchNum+=BATCH_SIZE
		#print(totalArr)
		return totalArr


# Credit: https://github.com/minsuk-heo/python_tutorial/blob/master/data_science/nlp/word2vec_tensorflow.ipynb
def createMap(words,sentences):
	word2int = {}

	for i,word in enumerate(words):
		word2int[word] = i

	WINDOW_SIZE = 2

	data = []
	for sentence in sentences:
		for idx,word in enumerate(sentence):
			for n in sentence[max(idx-WINDOW_SIZE,0) : min(idx+WINDOW_SIZE, len(sentence))+1]:
				if n != word:
					data.append([word,n])

	df = pd.DataFrame(data, columns = ['input', 'label'])
	return df,word2int




def to_one_hot_encoding(data_point_index,ONE_HOT_DIM):
    one_hot_encoding = np.zeros(ONE_HOT_DIM)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

def trainW2V(words,word2int,df):
	ONE_HOT_DIM = len(words)
	ITERATIONS = 3001 #20000

	X = []
	Y = []

	for x,y in zip(df['input'], df['label']):
		X.append(to_one_hot_encoding(word2int[x],ONE_HOT_DIM))
		Y.append(to_one_hot_encoding(word2int[x],ONE_HOT_DIM))


	X_train = np.asarray(X)
	Y_train = np.asarray(Y)

	x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
	y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))

	# Dim is 2 for visualization
	EMBEDDING_DIM = 2	


	w_dict = {
			'w1':tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM])),
			'w2':tf.Variable(tf.random_normal([EMBEDDING_DIM,ONE_HOT_DIM]))
	}

	b_dict = {
			'b1':tf.Variable(tf.random_normal([1])),
			'b2':tf.Variable(tf.random_normal([1]))
	}

	L1 = tf.add(tf.matmul(x,w_dict['w1']),b_dict['b1'])
	out = tf.nn.softmax(tf.add(tf.matmul(L1,w_dict['w2']),b_dict['b2']))
	pred = out

	loss = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(pred),axis=[1]))

	train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	iteration = ITERATIONS
	for i in tqdm(range(iteration)):
		lossTrain,_ = sess.run([loss,train_op], feed_dict = {x: X_train, y_label: Y_train})
		if i%3000==0:
			print('It',i,'loss is:',lossTrain)

	vectors = sess.run(w_dict['w1']+b_dict['b1'])
	#print(vectors)

	w2vDf = pd.DataFrame(vectors, columns = ['x1','x2'])
	w2vDf['word'] = words
	w2vDf = w2vDf[['word', 'x1', 'x2']]
	#print(w2vDf)

	'''
	fig, ax = plt.subplots()

	for word, x1, x2 in zip(w2vDf['word'],w2vDf['x1'],w2vDf['x2']):
		ax.annotate(word, (x1,x2))

	PADDING = 1.0
	x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
	y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
	x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
	y_axis_max = np.amax(vectors, axis=0)[1] + PADDING

	plt.xlim(x_axis_min,x_axis_max)
	plt.xlim(y_axis_min,y_axis_max)
	plt.rcParams['figure.figsize'] = (10,10)

	plt.show()
	'''
	embeddedDict = {}
	for word, x1, x2 in zip(w2vDf['word'],w2vDf['x1'],w2vDf['x2']):
		embeddedDict[word] = []
		embeddedDict[word].append(x1)
		embeddedDict[word].append(x2)
	embeddedDict['UNKOWNWORD'] = np.zeros(EMBEDDING_DIM)
	return embeddedDict






csvFileTrainw2v = CsvLoader('./ag_news_csv/train.csv')
uniqueWords,sentences = csvFileTrainw2v.getAllUniqueWords(10)
frame, word2code = createMap(uniqueWords,sentences)
embeddedDict = trainW2V(uniqueWords,word2code,frame)


csvFileTrain = CsvLoader('./ag_news_csv/train.csv')
inputWords = csvFileTrain.getData(embeddedDict)
print(embeddedDict.keys())
print(inputWords[0])
#inputWords = csvFileTrain.getData(embeddedDict)
#print(inputWords)



#codeDict = createCodesFromWords(uniqueWords,32)

#print(frame)






























