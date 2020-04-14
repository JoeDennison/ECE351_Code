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

steps = 1e-2
t = np.arange(0, 2+steps, steps)

step = 1e-2
omega = np.arange((10**3)+step, (10**6)+steps, steps)

R = 1000
L = 27*(10**-3)
C = 100*(10**-9)

num = [(1/(R*C)), 0]
den = [1,(1/(R*C)),(1/(L*C))]

t2b = signal.TransferFunction(num,den)
w , mag, phase = signal.bode(t2b, w=omega)

plt.figure(figsize = (10, 10))

plt.subplot(2,1,1)
plt.semilogx(w, mag)
plt.title("Task One: scipy.signal.bode")
plt.ylabel("Mag")

plt.subplot(2,1,2)
plt.semilogx(w, phase)
plt.ylabel("Phase")
plt.show()

#w, mag, phase = sig.bode()
#sys = con.TransferFunction(num, den)
#plt.figure(figsize = (10, 10))
#_ = con.bode(sys, omega, dB = True, Hz = True, deg = True, Plot = True)