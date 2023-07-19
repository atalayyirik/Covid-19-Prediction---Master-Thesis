// GDLL2.h
#pragma once

__declspec(dllexport) int InitGillespie(int, int, double, double);
__declspec(dllexport) double * RunGillespie(double, double, double, double);

//int InitGillespie(int RandS, int SisSize, double ParK1, double ParK2)
// double* RunGillespie(double TIni, double TMax, double Y01, double Y02)
