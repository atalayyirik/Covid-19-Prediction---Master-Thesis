#pragma once

//__declspec(dllexport) int InitGillespie(int, int, double, double);
//__declspec(dllexport) double * RunGillespie(double, double, double, double);

__declspec(dllexport) int InitKMC1(int, int, int, double, double, double, double, double);
__declspec(dllexport) double * RunKMC1(int, double, double, double, double);
__declspec(dllexport) int FreeMemoryKMC1(void);

