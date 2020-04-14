################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 8                                        #
# 3/30/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
import control as con
import pandas as pd

steps = 1e-2
t = np.arange(0, 20+steps, steps)
T = 8

def a(k):
	y = 0
	if k == 0:
		y = 2
	else:
		y = ((2*np.pi)/k)*np.sin(np.pi*k)
	return y

def b(k):
	y = 0
	y = (-2/(np.pi*k))*(np.cos(np.pi*k)-1)
	return y

def x(t,T,n):
	y = np.zeros(t.shape)
	w0 = (2*np.pi)/T
	y += 0.5*a(0)
	for i in range(1,n):
		y += a(i)*np.cos(i*w0*t)+b(i)*np.sin(i*w0*t)
	return y

a0 = a(0)
a1 = a(1)
b1 = b(1)
b2 = b(2)
b3 = b(3)

print("a(0):  ", a0)
print("a(1):  ", a1)
print("b(1):  ", b1)
print("b(2):  ", b2)
print("b(3):  ", b3)

n1 = x(t,T, 1)
n3 = x(t,T, 3)
n15 = x(t,T, 15)
n50 = x(t,T, 50)
n150 = x(t,T, 150)
n1500 = x(t,T, -1500)

plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, n1)
plt.title("Plots 1")
plt.ylabel('N = 1')

plt.subplot(3,1,2)
plt.plot(t, n3)
plt.ylabel("N = 3")

plt.subplot(3,1,3)
plt.plot(t, n15)
plt.ylabel('N = 15')
plt.show()

plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, n50)
plt.title("Plots 2")
plt.ylabel('N = 50')

plt.subplot(3,1,2)
plt.plot(t, n150)
plt.ylabel("N = 150")

plt.subplot(3,1,3)
plt.plot(t, n1500)
plt.ylabel('N = 1500')
plt.show()