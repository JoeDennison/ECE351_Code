################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 9                                        #
# 4/5/20                                       #
#                                              #
################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
import control as con
import pandas as pd

steps = 1e-2
t = np.arange(0, 2+steps, steps)

t1 = np.cos(2*np.pi*t)
fs = 200

def fftfunction(x,fs):
	N = len(x) 
	X_fft = sig.fftpack.fft(x) 
	X_fft_shifted = sig.fftpack.fftshift(X_fft)
	freq = np.arange(-N/2, N/2)*fs/N 
	X_mag = np.abs(X_fft_shifted)/N
	X_phi = np.angle(X_fft_shifted)
	for i in range(N):
		if(X_mag[i] < 1e-2):
			X_phi[i] = 0
		else:
			X_phi[i] = X_phi[i]
	return freq, X_mag , X_phi


#plt.stem(freq, X_mag) # you will need to use stem to get these plots to be
#plt.stem(freq, X_phi) # correct, remember to label all plots appropriately
							
t1freq, t1mag, t1phi = fftfunction(t1, fs)		
plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, t1)
plt.title("Task 4a: Cos(2*pi*t)")
plt.ylabel('Original')
plt.xlabel('time domain')

plt.subplot(3,1,2)
plt.stem(t1freq, t1mag)
plt.xlim(left = -2)
plt.xlim(right = 2)
plt.ylabel('Mag')
plt.xlabel('freq domain')

plt.subplot(3,1,3)
plt.stem(t1freq, t1phi)
plt.xlim(left = -2)
plt.xlim(right = 2)
plt.ylabel('Phase')
plt.xlabel('freq domain')

t2 = 5*np.sin(2*np.pi*t)
t2freq, t2mag, t2phi = fftfunction(t2, fs)		
plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, t2)
plt.title("Task 4b: 5*sin(2*pi*t)")
plt.ylabel('Original')
plt.xlabel('time domain')

plt.subplot(3,1,2)
plt.stem(t2freq, t2mag)
plt.xlim(left = -2)
plt.xlim(right = 2)
plt.ylabel('Mag')
plt.xlabel('freq domain')

plt.subplot(3,1,3)
plt.stem(t2freq, t2phi)
plt.xlim(left = -3)
plt.xlim(right = 3)
plt.ylabel('Phase')
plt.xlabel('freq domain')

t3 = (2*np.cos((2*np.pi*2*t)-2))+(np.sin((2*np.pi*6*t)+3)**2)
t3freq, t3mag, t3phi = fftfunction(t3, fs)		
plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t, t3)
plt.title("Task 4c: (2*np.cos((2*np.pi*2*t)-2))+(np.sin((2*np.pi*6*t)+3)**2)")
plt.ylabel('Original')
plt.xlabel('time domain')

plt.subplot(3,1,2)
plt.stem(t3freq, t3mag)
plt.xlim(left = -15)
plt.xlim(right = 15)
plt.ylabel('Mag')
plt.xlabel('freq domain')

plt.subplot(3,1,3)
plt.stem(t3freq, t3phi)
plt.xlim(left = -15)
plt.xlim(right = 15)
plt.ylabel('Phase')
plt.xlabel('freq domain')

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


t_2 = np.arange(0, 16+steps, steps)
t5 = x(t_2, T, 15)
t5freq, t5mag, t5phi = fftfunction(t5, fs)		
plt.figure(figsize = (10, 10))

plt.subplot(3,1,1)
plt.plot(t_2, t5)
plt.title("Task Five: Fourier Series")
plt.ylabel('Original')
plt.xlabel('time domain')

plt.subplot(3,1,2)
plt.stem(t5freq, t5mag)
plt.xlim(left = -2)
plt.xlim(right = 2)
plt.ylabel('Mag')
plt.xlabel('freq domain')

plt.subplot(3,1,3)
plt.stem(t5freq, t5phi)
plt.xlim(left = -2)
plt.xlim(right = 2)
plt.ylabel('Phase')
plt.xlabel('freq domain')