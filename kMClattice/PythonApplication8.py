import numpy as np
import scipy
from statistics import mean
from statistics import stdev 
from time import time

import ctypes 

# indicate the correct path to dll !!!!!!!!!!
dll = ctypes.CDLL("GDLL2.dll") 


#int InitGillespie(int RandS, int SisSize, double ParK1, double ParK2)
dll.InitGillespie.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double] 
dll.InitGillespie.restype = ctypes.c_int 
dll.InitGillespie(1, 1024, 1.0, 1.0)
#print("init ok")

NPaths = 100000
y = ctypes.POINTER(ctypes.c_double * 4)()
Theta0 = np.zeros(NPaths)
Theta1 = np.zeros(NPaths)
Theta2 = np.zeros(NPaths)
Time = np.zeros(NPaths)
Y0_1 = 0.02
Y0_2 = 0.0
tmax = 2.0
# double* RunGillespie(double TIni, double TMax, double Y01, double Y02)
t0 = time()
dll.RunGillespie.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
dll.RunGillespie.restype = ctypes.POINTER(ctypes.c_double * 4)
for NP in range(NPaths):
   y = dll.RunGillespie(0.0, tmax, Y0_1, Y0_2).contents
   #print(y[0], y[1], y[2], y[3]) 
   Theta0[NP] = y[0]; 
   Theta1[NP] = y[1]; 
   Theta2[NP] = y[2]; 
   Time[NP] = y[3];

print(f"SDE sampling took {time()-t0} seconds for {NPaths} paths.")
print("Mean0 = ", mean(Theta0[:]), "  SD0 = ", stdev(Theta0[:])) 
print("Mean1 = ", mean(Theta1[:]), "  SD1 = ", stdev(Theta1[:])) 
print("Mean2 = ", mean(Theta2[:]), "  SD2 = ", stdev(Theta2[:])) 
print("MeanTime = ", mean(Time[:]))