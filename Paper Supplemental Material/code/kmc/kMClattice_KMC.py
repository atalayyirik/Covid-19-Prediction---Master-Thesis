import numpy as np
import scipy
from statistics import mean
from statistics import stdev 
from time import time

import ctypes 


class SIR_KMC:
    """
    Python wrapper for the DLL of the SIR Gillespie simulation.
    """
    #  int InitKMC1(int RandSeed, int LatN1, int LatN2, double ParK1, double ParK2, double ParK3, double ParDif1, double ParDif2)

    def __init__(self, RandS = 1, N1 = 32, N2 = 32, k1 = 1.0, k2 = 1.0, k3 = 0.0, Dif1 = 1.0, Dif2 = 1.0, random_state = 1, GDLL_path="C:\Python\KMCDLL1.dll"):
        self.RandS = RandS
        self.N1 = N1
        self.N2 = N2
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
        dll = ctypes.CDLL(GDLL_path)
        
        # initialize the dll
        dll.InitKMC1.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
        dll.InitKMC1.restype = ctypes.c_int 
        # int RandS, int SisSize, double ParK1, double ParK2
        ErrInit = dll.InitKMC1(RandS, N1, N2, k1, k2, k3, Dif1, Dif2)
        if (ErrInit == 0):
            print("init-KMC, ok")
        else:
            print("error1-KMC")
            exit(1)

        self._dll = dll
        
    def simulate(self, y0, time_max=4, step_size=1e-2, Point1 = 1):
        NPaths = y0.shape[0]

        y = ctypes.POINTER(ctypes.c_double * 4)()
        Thetas = []
        Times = []
        
        self._dll.RunKMC1.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
        self._dll.RunKMC1.restype = ctypes.POINTER(ctypes.c_double * 4)

        for i in range(NPaths):
            Y0_1 = y0[i,0]
            Y0_2 = y0[i,1]
            current_time = 0.0
            current_path = []
            current_times = []
            k = 1
            while current_time < time_max:

                if (current_time == 0.0): 
                    y = self._dll.RunKMC1(0, current_time, k*step_size, Y0_1, Y0_2).contents
                else:
                    y = self._dll.RunKMC1(1, current_time, k*step_size, 0.0, 0.0).contents;  # use the previous lattice 

                current_time = y[3]
                Y0_1, Y0_2 = y[1], y[2]

                if k >= Point1:
                    current_path.append([y[0], y[1], y[2]])
                    current_times.append(current_time)

                k += 1

            if current_path != []:
               Thetas.append(np.row_stack(current_path).reshape(-1,3))
               Times.append(np.array(current_times).ravel())
        return Times, Thetas