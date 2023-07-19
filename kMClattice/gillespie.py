import numpy as np
import scipy
from statistics import mean
from statistics import stdev 
from time import time

import ctypes 


class SIRGillespie:
    """
    Python wrapper for the DLL of the SIR Gillespie simulation.
    """
    
    def __init__(self, RandS = 1, N = 1024, k1 = 1.0, k2 = 1.0, random_state = 1, GDLL_path = "./GDLL2.dll"):
        self.RandS = RandS
        self.N = N
        self.k1 = k1
        self.k2 = k2
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)
        
        dll = ctypes.CDLL(GDLL_path)
        
        # initialize the dll
        dll.InitGillespie.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double] 
        dll.InitGillespie.restype = ctypes.c_int 
        # int RandS, int SisSize, double ParK1, double ParK2
        dll.InitGillespie(RandS, N, k1, k2)
        self._dll = dll
        
    def simulate(self, y0, time_max=4, step_size=1e-2):
        NPaths = y0.shape[0]
        y = ctypes.POINTER(ctypes.c_double * 4)()
        Thetas = []
        Times = []
        # double* RunGillespie(double TIni, double TMax, double Y01, double Y02)
        self._dll.RunGillespie.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
        self._dll.RunGillespie.restype = ctypes.POINTER(ctypes.c_double * 4)
        for i in range(NPaths):
            Y0_1 = y0[i,0]
            Y0_2 = y0[i,1]
            current_time = 0.0
            current_path = []
            current_times = []
            k = 1
            while current_time < time_max:
                y = self._dll.RunGillespie(current_time, k*step_size, Y0_1, Y0_2).contents
                current_time = y[3]
                Y0_1, Y0_2 = y[1], y[2]
                current_path.append([y[0], y[1], y[2]])
                current_times.append(current_time)
                k += 1
            Thetas.append(np.row_stack(current_path).reshape(-1,3))
            Times.append(np.array(current_times).ravel())
        return Times, Thetas