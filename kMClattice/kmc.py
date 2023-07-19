import numpy as np
import scipy
from statistics import mean
from statistics import stdev 
from time import time

import ctypes 


class SIRKMCLattice:
    """
    Python wrapper for the DLL of the SIR kMC lattice simulation.
    """
    
    def __init__(self, random_state = 1,
                 N1 = 32, N2 = 32,
                 k1 = 1.0, k2 = 1.0, k3 = 0.0,
                 dif1 = 100.0, dif2 = 100.0,
                 KMCDLL_path = "./KMCDLL1.dll"):
        self.random_state = int(random_state)
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.dif1 = dif1
        self.dif2 = dif2
        self.rng = np.random.default_rng(self.random_state)
        
        dll = ctypes.CDLL(KMCDLL_path)
        
        # initialize the dll
        dll.InitKMC1.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double] 
        dll.InitKMC1.restype = ctypes.c_int
        
        # int InitKMC1(int RandSeed, int LatN1, int LatN2, double ParK1, double ParK2, double ParK3, double ParDif1, double ParDif2)
        ErrInit = dll.InitKMC1(self.random_state,
                               self.N1, self.N2,
                               self.k1, self.k2, self.k3,
                               self.dif1, self.dif2)
        if not(ErrInit == 0):
            raise ValueError("Initialization not ok.")
        self._dll = dll
        
    def simulate(self, y0, time_max=4, step_size=1e-2):
        NPaths = y0.shape[0]
        y = ctypes.POINTER(ctypes.c_double * 4)()
        Thetas = []
        Times = []
        # double* RunGillespie(double TIni, double TMax, double Y01, double Y02)
        self._dll.RunKMC1.argtypes = [ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, ctypes.c_double] 
        self._dll.RunKMC1.restype = ctypes.POINTER(ctypes.c_double * 4)
        for i in range(NPaths):
            Y0_1 = y0[i,0]
            Y0_2 = y0[i,1]
            INIT_LATTICE = 0
            USE_LATTICE = 1
            current_time = 0.0
            current_path = []
            current_times = []
            while current_time < time_max:
                if current_time == 0.0:
                    y = self._dll.RunKMC1(INIT_LATTICE, current_time, step_size, Y0_1, Y0_2).contents
                else:
                    y = self._dll.RunKMC1(USE_LATTICE, current_time, current_time+step_size, 0.0, 0.0).contents
                current_time = y[3]
                Y0_1, Y0_2 = y[1], y[2]
                current_path.append([y[0], y[1], y[2]])
                current_times.append(current_time)
            Thetas.append(np.row_stack(current_path).reshape(-1,3))
            Times.append(np.array(current_times).ravel())
        return Times, Thetas