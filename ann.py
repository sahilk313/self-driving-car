import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import numpy as np 
import random 
import cv2
import math 
from scipy.special import expit 

filelocation = name = raw_input("Enter the location where images and data.txt are stored eg. ./steering ")
alpha = 0.01
batch=64
epochs = 1000

size=21999
examples = int(0.8*size)
dropout = 0
  
def rearrange(p,q):
		a = np.arange(len(p))
		np.random.shuffle(a)
		ret1 = p[a]
		ret2 = q[a]
		return ret1,ret2

def wof(n1):
	size = n1.shape;
	w = np.linspace(-0.01,0.01,size[0]*(size[1]-1),dtype=np.float64)
	w.resize((size[0],(size[1]-1)))
	b=np.zeros((size[0],1))
	n1 = np.append(b,w,axis=1)
	return n1		

def sigmoid(x):
	return expit(x)

def gradient(hid):
	ans = size
	ans = ans + 4
	ans2 = 0
	if(ans==examples):
		ans2 = ans
	anss =hid*(1-hid)
	return anss 		

class myClass:
	def __init__(self, x, y,b):
		self.w1 = wof(np.ndarray(shape=(512,1025)))
		self.w2 = wof(np.ndarray(shape=(64,513)))
		self.w3 = wof(np.ndarray(shape=(1,65)))
		self.X = x
		self.y = y
		self.batch = b
		self.out = np.zeros(self.y.shape)
	
	def forwardprop(self):
		self.out = np.zeros(self.y.shape)			
		self.hid1 = sigmoid(np.matmul(self.X, self.w1.T))
		bias = np.ones((len(self.hid1),1))		
		self.hid1 = np.append(bias,self.hid1,axis=1)		
		self.hid2 = sigmoid(np.matmul(self.hid1, self.w2.T))
		bias = np.ones((len(self.hid2),1))		
		self.hid2 = np.append(bias,self.hid2,axis=1)		
		self.out =	np.matmul(self.hid2, self.w3.T)		
	
	def backwardprop(self):
		v_ = self.out - self.y
		v = np.matmul(v_.T,self.hid2)		
		w2_ = np.matmul(v_,self.w3)*gradient(self.hid2)
		w2_ = w2_[:,1:]
		w21 = np.matmul(w2_.T,self.hid1)		
		w1_ =	np.matmul(w2_,self.w2)*gradient(self.hid1)
		w1_ =  w1_[:,1:]
		w11 = np.matmul(w1_.T,self.X)		
		self.w1 = self.w1 - alpha*w11/self.batch
		self.w2 = self.w2 - alpha*w21/self.batch
		self.w3 = self.w3 - alpha*v/self.batch
	
	def error(self):
		err = np.sum(np.square(self.out - self.y))/(2*size)
		return err	

total = list(range(0, size))
training_set = random.sample(total,examples)
training_dict = {}
for item in training_set:
	training_dict[item] = 1
testing_set = []
for i in range(0,size):
	if(not i in training_dict):
		testing_set.append(i)
	
training = np.ndarray(shape=(1,1024))
testing = np.ndarray(shape=(1,1024)) 

training = np.delete(training, (0), axis=0)
testing = np.delete(testing, (0), axis=0)

for i in training_set:
	name = ""
	name = filelocation+'/img_'+str(i)+'.jpg'
	img = cv2.imread(name,cv2.IMREAD_GRAYSCALE) 
	b = img.ravel()
	mean = np.mean(b)
	mini = np.amin(b)
	maxi = np.amax(b)
	b=b-mean
	b=b/(maxi-mini)		
	training = np.vstack([training, b])

for i in testing_set:
	name = ""
	name = filelocation+'/img_'+str(i)+'.jpg'
	img = cv2.imread(name,cv2.IMREAD_GRAYSCALE) 
	b = img.ravel()
	mean = np.mean(b)
	mini = np.amin(b)
	maxi = np.amax(b)
	b=b-mean
	b=b/(maxi-mini)		
	testing = np.vstack([testing, b])
  
bias = np.ones((len(training),1))		
training = np.append(bias,training,axis=1)
bias = np.ones((len(testing),1))		
testing = np.append(bias,testing,axis=1)
file_ = open(filelocation+'/data.txt', 'r+')
data = []
for line in file_.readlines():
	parts = line.split()
	data.append(float(parts[1]))
Y = np.delete(np.ndarray(shape=(1,1)), (0), axis=0)	
Ytest = np.delete(np.ndarray(shape=(1,1)), (0), axis=0)	
for i in training_set:
	Y = np.vstack([Y, data[i]])
for i in testing_set:
	Ytest = np.vstack([Ytest, data[i]])

myObj = myClass(training,Y,len(Y))

dim=int(size/batch)+1
lefti=0
righti=0

plottrain = []
xaxi = []
plottest = []
xaxii = []
for i in range(epochs):		
	training,Y = rearrange(training,Y)
	testing,Ytest = rearrange(testing,Ytest)
	for j in range(dim):
		lefti = batch*j
		righti = batch*(j+1)				
		if(righti>=size or lefti>righti):
		   break;				   
		training_batch = training[lefti:righti]
		Y_batch = Y[lefti:righti]				
		myObj.X=training_batch
		myObj.y= Y_batch		
		myObj.forwardprop()
		myObj.backwardprop()	
	myObj.X=training
	myObj.y= Y
	myObj.forwardprop()
	print(myObj.error())
	plottrain.append(myObj.error())
	xaxi.append(i)
	myObj.X = testing
	myObj.y = Ytest
	myObj.forwardprop()
	print('Test error: ')
	print(myObj.error())	
	plottest.append(myObj.error())
	xaxii.append(i)

plt.plot(xaxi, plottrain)
plt.show()