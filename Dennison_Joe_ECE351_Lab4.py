################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 0                                        #
# 1/21/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
import control as con
import pandas as pd
#%% Part one
steps = 1e-2
t = np.arange(-10 +steps,10 + steps, steps)

def u(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		if (t[i] > 0):
			y[i] = 1;
		else:
			y[i] = 0;
	return y

def r(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		if (t[i] > 0):
			y[i] = t[i]
		else:
			y[i] = 0
	return y

def h_1(t):
	y=np.zeros(t.shape)
	y=np.exp(2*t)*u(1-t)
	return y

def h_2(t):
	y=np.zeros(t.shape)
	y=u(t-2)-u(t-6)
	return y

def h_3(t):
	y=np.zeros(t.shape)
	y=np.cos(2*np.pi*.25*t)*u(t)
	return y

h1= h_1(t)
h2= h_2(t)
h3= h_3(t)

plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, h1)
plt.title('Part One')
plt.ylabel('h1')

plt.subplot(3,1,2)
plt.plot(t, h2)
plt.ylabel('h2')

plt.subplot(3,1,3)
plt.plot(t, h3)
plt.ylabel('h3')
plt.show()
#%% Part two
def my_conv(f1,f2):
	Nf1= len(f1)
	Nf2= len(f2)
	f1Extended= np.append(f1, np.zeros((1,Nf2-1)))
	f2Extended= np.append(f2, np.zeros((1,Nf1-1)))
	result = np.zeros(f1Extended.shape)
	for i in range(Nf1+Nf2-2):
		result[i] = 0
		for j in range(Nf1):
			if(i-j+1 >0):
				try:
					result[i] += f1Extended[j]*f2Extended[i-j+1]
				except:
					print(i,j)
	return result

steps = 1e-2
t = np.arange(-10,10 + steps, steps)

y1= h_1(t)
y2= h_2(t)
y3= h_3(t)
y4= u(t)

NN= len(t)
tExtended = np.arange(2*t[0],2*t[NN-1] + steps, steps)

conv14= my_conv(y1, y4)*steps
conv24= my_conv(y2, y4)*steps
conv34= my_conv(y3, y4)*steps

plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(tExtended, conv14)
plt.title('Part Two')
plt.ylabel('h1(t)*u(t)')

plt.subplot(3,1,2)
plt.plot(tExtended, conv24)
plt.ylabel('h2(t)*u(t)')

plt.subplot(3,1,3)
plt.plot(tExtended, conv34)
plt.ylabel('h3(t)*u(t)')
plt.show()

#%% Part two b

def hand1(t):
	y=np.zeros(t.shape)
	y=(.5*np.exp(2*t)*u(1-t))+(.5*np.exp(2)*u(t-1))
	return y

def hand2(t):
	y=np.zeros(t.shape)
	y=r(t-2)-r(t-6)
	return y

def hand3(t):
	y=np.zeros(t.shape)
	y= (2/np.pi)*np.sin(.5*np.pi*t)*u(t)
	return y

y5= hand1(t)
y6= hand2(t)
y7= hand3(t)

plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, y5)
plt.title('Part Two b')
plt.ylabel('h1(t)*u(t)')

plt.subplot(3,1,2)
plt.plot(t, y6)
plt.ylabel('h2(t)*u(t)')

plt.subplot(3,1,3)
plt.plot(t,y7)
plt.ylabel('h3(t)*u(t)')
plt.show()