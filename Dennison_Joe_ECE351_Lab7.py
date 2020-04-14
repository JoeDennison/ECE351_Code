################################################
#                                              #
# Joseph Dennison                              #
# ECE 351-51                                   #
# Lab 7                                        #
# 3/10/20                                      #
#                                              #
#################################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import control as con
import pandas as pd

steps = 1e-2
t = np.arange(0, 4+steps, steps)

Gnum = [1, 9]
Gden = [1, -2, -40, -64]

Anum = [1, 4]
Aden = [1, 4, 3]

B = [1, 26, 168]

z1, p1, k1 = sig.tf2zpk(Gnum, Gden)
z2, p2, k2 = sig.tf2zpk(Anum, Aden)
broot = np.roots(B)

print("Z1:  ",z1)
print("P1:  ",p1)
print("K1:  ",k1)

print("Z2:  ",z2)
print("P2:  ",p2)
print("K2:  ",k2)

print("Broots:  ",broot)

opennum = sig.convolve(Gnum, Anum)
openden = sig.convolve(Gden, Aden)
tout1, yout1 = sig.step((opennum, openden), T=t)

plt.figure(figsize = (10, 10))

plt.plot(tout1, yout1)
plt.title('Part One: Open System')

closenum= sig.convolve(Anum, Gnum)
#den1= sig.convolve(Gnum, B)
#print(len(den1))
#print(len(Gden)) 
#den2 = den1 + Gden
#closeden= sig.convolve(Aden, den2)
closeden= sig.convolve(Aden, (sig.convolve(Gnum, B)+Gden))

tout2, yout2 = sig.step((closenum, closeden), T=t)

plt.figure(figsize = (10, 10))

plt.plot(tout2, yout2)
plt.title('Part Two: Closed System')