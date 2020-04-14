################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 2                                        #
# 2/4/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
import control as con
import pandas as pd

#%%Example Code
plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(0,5 + steps, steps)

print ('Number of elements: len(t) = ' ,len(t), '\nFirstElement: t[0] = ', t[0], 
   '\nLast Element: t[len(t) -1] = ', t[len(t)-1])

def example1(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		if i < (len(t) + 1)/3:
			y[i] = t[i]**2
		else:
			y[i] = np.sin(5*t[i]) + 2
	
	return y

y = example1(t)

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('Background - Illustration of for Loops and if/else Statements')

t = np.arange(0, 5 + 0.25, 0.25)
y = example1(t)

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Poor Resolution')
plt.xlabel('t')
plt.show()

#%%Part 1
plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(0,10 + steps, steps)

print ('Number of elements: len(t) = ' ,len(t), '\nFirstElement: t[0] = ', t[0], 
   '\nLast Element: t[len(t) -1] = ', t[len(t)-1])

def func1(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		y[i] = np.cos(t[i])
		
	return y

y = func1(t)

plt.figure(figsize = (10,7))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.ylabel('y(t) with Good Resolution')
plt.title('y = cos(t) using the Example Code')

#t = np.arange(0, 5 + 0.25, 0.25)
#y = example1(t)

#plt.subplot(2, 1, 2)
#plt.plot(t, y)
#plt.grid()
#plt.ylabel('y(t) with Poor Resolution')
#plt.xlabel('t')
#plt.show()

#%%Part 2
plt.rcParams.update({'font.size': 14})

steps = 1e-2
t = np.arange(-5 +steps,10 + steps, steps)

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

def f(t):
	y = np.zeros(t.shape)
	y = r(t)+5*u(t-3)-r(t-3)-2*u(t-6)-2*r(t-6)
	return y


y = f(t)

plt.figure(figsize = (10,16))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.title('Part two: recreating plot from handout')

#%%Part 3

t = np.arange(-5 +steps,10 + steps, steps)

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

def f(t):
	y = np.zeros(t.shape)
	y = r(t)+5*u(t-3)-r(t-3)-2*u(t-6)-2*r(t-6)
	return y

y = f(2*t)

plt.figure(figsize = (10,16))
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.grid()
plt.title('f(2t) of part two')

y = f(t)
dt = np.diff(t)
dy = np.diff(y, axis = 0)/dt
plt.figure(figsize=(10,7))
plt.plot(t, y, '--', label = 'y(t)')
plt.plot(t[range(len(dy))], dy[:0], label = 'dy(t)/dt')
plt.title('Derivative WRT time')
plt.legend()
plt.grid()
plt.ylim([-2,10])
plt.show()