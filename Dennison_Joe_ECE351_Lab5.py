################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 5                                        #
# 2/25/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con
import pandas as pd

def u(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		if (t[i] > 0):
			y[i] = 1;
		else:
			y[i] = 0;
	return y

R1 = 1000
L1 = 27*10**(-3)
C1 = 100*10**(-9)
steps = 1e-6
t = np.arange(0, 1.2e-3+steps, steps)

num = [0, (1/(C1*R1)), 0]
den = [1, (1/(C1*R1)), (1/np.sqrt(L1*C1))**2]

tout, yout = sig.impulse((num,den), T = t)



def sinemethod(R, L, C, t):
	alpha= (-1/(2*R*C))
	omega= (.5*np.sqrt(((1/(C*R))**2-4*(1/(np.sqrt(L*C)))**2)+0*1j))
	p= alpha + omega
	g= (1/(R*C)*p)
	g_mag= np.abs(g)
	g_rad= np.angle(g)
	#g_deg= (g_rad*180)/np.pi
	y= (g_mag/np.abs(omega)*np.exp(alpha*t)*np.sin(np.abs(omega)*t+g_rad)*u(t))
	return y

y1= sinemethod(R1, L1, C1, t)

plt.figure(figsize = (10, 10))


plt.subplot(2,1,1)
plt.plot(t, y1)
plt.title('Part One: Hand calculations')

plt.subplot(2,1,2)
plt.plot(tout, yout)
plt.title('Part One: signal impulse')

tout2, yout2 = sig.step((num, den), T=t)

plt.figure(figsize = (10, 5))

plt.subplot(1,1,1)
plt.plot(tout2, yout2)
plt.title('Part Two: sig step function')

