################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 10                                       #
# 4/13/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy as sig
from scipy import signal
import control as con
import pandas as pd

steps = 1e2

omega = np.arange(1e3, 1e6+steps, steps)

R = 1000
L = 27e-3
C = 100e-9

mag1= 20*np.log10((omega/(R*C))/np.sqrt(omega**4+(((1/(R*C))**2)-2/(L*C))*omega**2+(1/(L*C))**2))
#mag1 = 20*np.log10((omega/(R*C))/np.sqrt((omega**4)+ ((1/(L*C)**2)+((omega**2)*(((1/(R*C))**2)-2(1/(L*C)))))))
phase1 = ((np.pi/2)-np.arctan((omega/(R*C))/(-omega**2 + 1/(L*C)))) * 180/np.pi

for i in range(len(phase1)):
	if(phase1[i] > 90):
		phase1[i] = (phase1[i]-180)

plt.figure(figsize = (10, 10))
plt.subplot(2,1,1)
plt.semilogx(omega, mag1)
plt.title("Task One: Prelab")
plt.ylabel("Mag")

plt.subplot(2,1,2)
plt.semilogx(omega,phase1)
plt.ylabel("Phase")

num = [(1/(R*C)), 0]
den = [1,(1/(R*C)),(1/(L*C))]

w2 , mag2, phase2 = signal.bode((num,den), omega)

plt.figure(figsize = (10, 10))

plt.subplot(2,1,1)
plt.semilogx(w2, mag2)
plt.title("Task One: scipy.signal.bode")
plt.ylabel("Mag")

plt.subplot(2,1,2)
plt.semilogx(w2, phase2)
plt.ylabel("Phase")
plt.show()

sys = con.TransferFunction(num,den)
_ = con.bode(sys, omega, dB= True, deg = True, Plot = True)


steps2 = 1e-9
t2 = np.arange(0, 0.01+steps2, steps2)
x = (np.cos(2*np.pi*100*t2)+np.cos(2*np.pi*3024*t2) + np.sin(2*np.pi*5000*t2))

num2, den2, = signal.bilinear(num, den, 1/steps2)
filtered = signal.lfilter(num2,den2,x)
plt.figure(figsize = (10,10))
plt.subplot(2,1,1)
plt.plot(t2,x)
plt.title("Part two: bilinear and lfilter")
plt.ylabel("Unfiltered")

plt.subplot(2,1,2)
plt.plot(t2, filtered)
plt.ylabel("filtered")
plt.show()