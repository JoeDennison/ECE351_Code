################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 6                                        #
# 3/3/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con
import pandas as pd

steps = 1e-2
t = np.arange(0, 2+steps, steps)


def u(t):
	y = np.zeros(t.shape)
	
	for i in range(len(t)):
		if (t[i] > 0):
			y[i] = 1;
		else:
			y[i] = 0;
	return y

# Part One
prelab = np.exp(-6*t)*u(t)+(-.5*np.exp(-4*t)*u(t))+(.5*u(t))



num = [1, 6, 12]
den = [1, 10, 24]
tout1, yout1 = sig.step((num, den), T=t)

plt.figure(figsize = (10, 10))
plt.subplot(2,1,1)
plt.plot(t, prelab)
plt.title('Part One: Hand calculations')

plt.subplot(2,1,2)
plt.plot(tout1, yout1)
plt.title('Part One: Builtin step')

plt.show()

num2 = [0, 1, 6, 12]
den2 = [1, 10, 24, 0]


r,p,k = sig.residue(num2, den2)
print("Part One: Prelab")
print("R:  ", r)
print("P:  ", p)
print("K:  ", k)

num3 = [25250]
den3 = [1, 18, 218, 2036, 9085, 25250, 0]
r1, p1, k1 = sig.residue(num3, den3)
print("Part Two: Difficult by Hand")
print("R:  ",r1)
print("P:  ",p1)
print("K:  ",k1)

# Part Two

t = np.arange(0, 4.5+steps, steps)

def cosinemethod(r,p,t):
    y = np.zeros(t.shape)    
    for i in range(len(r)):
        alpha = np.real(p[i]) 
        omega = np.imag(p[i])        		            
        kmag = np.abs(r[i]) 
        kphase = np.angle(r[i])           
        y += (kmag*np.exp(alpha*t)*np.cos(omega*t + kphase))*u(t)        
    return y

cos1 = cosinemethod(r1, p1, t)

num4 = [0, 0, 0, 0, 0, 25250]
den4 = [1, 18, 218, 2036, 9085, 25250]
tout2, yout2 = sig.step((num4, den4), T=t)
plt.figure(figsize = (10, 10))

plt.subplot(2,1,1)
plt.plot(t, cos1)
plt.title('Part Two: cosinemethod')

plt.subplot(2,1,2)
plt.plot(tout2, yout2)
plt.title('Part Two: Builtin step')