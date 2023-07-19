import numpy as np
import scipy
from statistics import mean
from statistics import stdev 
from time import time

import ctypes 

dll = ctypes.CDLL("C:\Python\KMCDLL1.dll") 
#  int InitKMC1(int RandSeed, int LatN1, int LatN2, double ParK1, double ParK2, double ParK3, double ParDif1, double ParDif2)
dll.InitKMC1.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
dll.InitKMC1.restype = ctypes.c_int 
# Lattice 32*32, D=100
ErrInit = dll.InitKMC1(1, 32, 32, 1.0, 1.0, 0.0, 100.0, 100.0)
if (ErrInit == 0):
  print("init1, ok")
else:
  exit(1)

# Example1: 1000 copies at t=2
NPaths = 1000
tmax = 2.0
Y0_1 = 0.02
Y0_2 = 0.0
y = ctypes.POINTER(ctypes.c_double * 4)
Theta0 = np.zeros(NPaths)
Theta1 = np.zeros(NPaths)
Theta2 = np.zeros(NPaths)
Time = np.zeros(NPaths)
#  double*  RunKMC1(int RunType, double TIni, double TMax, double Y01, double Y02)
t0 = time()
dll.RunKMC1.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
dll.RunKMC1.restype = ctypes.POINTER(ctypes.c_double * 4)
for NP in range(NPaths):
   y = dll.RunKMC1(0, 0.0, tmax, Y0_1, Y0_2).contents
   #print(y[0], y[1], y[2], y[3]) 
   Theta0[NP] = y[0]; 
   Theta1[NP] = y[1]; 
   Theta2[NP] = y[2]; 
   Time[NP] = y[3];

print(f"KMC sampling took {time()-t0} seconds for {NPaths} paths.")
print("Mean0 = ", mean(Theta0[:]), "  SD0 = ", stdev(Theta0[:])) 
print("Mean1 = ", mean(Theta1[:]), "  SD1 = ", stdev(Theta1[:])) 
print("Mean2 = ", mean(Theta2[:]), "  SD2 = ", stdev(Theta2[:])) 
print("MeanTime = ", mean(Time[:]))
print()

#
# Example2: one copy, 100*100 lattice, t=[0,4]; the results with dt=0.2 are printed (lattice is not updated during a new call of RunKMC1) 
dll.FreeMemoryKMC1()

ErrInit = dll.InitKMC1(1, 100, 100, 1.0, 1.0, 0.0, 100.0, 100.0)
if (ErrInit == 0):
  print("init2, ok")
else:
  exit(1)

tstep = 0.2
tcur = 0.0
tnext = 0
tmax = 4.0
I = 0

while (tcur < tmax):
	if (tcur == 0.0): 
		y = dll.RunKMC1(0, tcur, tstep, Y0_1, Y0_2).contents # initialize lattice configuration for given Y0_1, Y0_2
	else:
		y = dll.RunKMC1(1, tcur, tnext, 0.0, 0.0).contents;  # use the previous lattice 
	I = I + 1;
	tcur = y[3]
	tnext = I*tstep
	print("{:.2f}".format(tcur), "{:.4f}".format(y[0]), "{:.4f}".format(y[1]), "{:.4f}".format(y[2])) 

