################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 3                                        #
# 2/18/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
import control as con
import pandas as pd
#%% Part1
steps = 1e-2
t = np.arange(0 +steps,20 + steps, steps)

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

def f_1(t):
	y = np.zeros(t.shape)
	y = u(t-2)-u(t-9)
	return y


def f_2(t):
	y = np.zeros(t.shape)
	y = np.exp(-t)*u(t)
	return y

def f_3(t):
	y = np.zeros(t.shape)
	y = r(t-2)*(u(t-2)-u(t-3))+r(4-t)*(u(t-3)-u(t-4))
	return y

y1= f_1(t)
y2= f_2(t)
y3= f_3(t)

plt.figure(figsize = (10,10))
plt.subplot(3, 1, 1)
plt.plot(t, y1)
plt.grid()
plt.title('Plot of f1, f2, f3')
plt.xlabel('t [s]')
plt.ylabel('f1(t)')


plt.figure(figsize = (10,10))
plt.subplot(3,1,2)
plt.plot(t,y2)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('f2(t)')

plt.figure(figsize = (10,10))
plt.subplot(3,1,3)
plt.plot(t,y3)
plt.grid()
plt.xlabel('t [s]')
plt.ylabel('f3(t)')
#%% Part 2
#Convolve
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
					result[i] += f1Extended[i]*f2Extended[i-j+1]
				except:
					print(i,j)
	return result

steps = 1e-2
t = np.arange(0, 20 + steps, steps)
NN= len(t)
tExtended = np.arange(0,2*t[NN-1], steps)

f1 = f_1(t)
f2 = f_2(t)
f3 = f_3(t)

conv12= my_conv(f1, f2)*steps
conv12check=sig.convolve(f1,f2)*steps

conv23= my_conv(f2, f3)*steps
conv23check=sig.convolve(f2,f3)*steps

conv13= my_conv(f1, f3)*steps
conv13check= sig.convolve(f1,f3)*steps

plt.figure(figsize = (10, 10))
plt.subplot(3, 1, 1)
plt.plot(tExtended, conv12, label = 'User-Defined Convolution')
plt.plot(tExtended, conv12check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_1(t) * f_2(t)')
plt.title('Convolution of f_1 and f_2')
plt.show()

plt.figure(figsize = (10, 10))
plt.subplot(3,1,2)
plt.plot(tExtended, conv23, label = 'User-Defined Convolution')
plt.plot(tExtended, conv23check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_2(t) * f_3(t)')
plt.title('Convolution of f_2 and f_3')
plt.show()

plt.figure(figsize = (10, 10))
plt.subplot(3,1,3)
plt.plot(tExtended, conv13, label = 'User-Defined Convolution')
plt.plot(tExtended, conv13check, '--', label = 'Built-In Convolution')
plt.ylim([0, 1.2])
plt.grid()
plt.legend()
plt.xlabel('t [s]')
plt.ylabel('f_1(t) * f_3(t)')
plt.title('Convolution of f_1 and f_3')
plt.show()
