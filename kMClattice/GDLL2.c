#include "Windows.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "GDLL2.h"


typedef unsigned long long int ulong;
typedef unsigned int uint;

uint N;
double CN, K1, K2;
int ErrCode = 0;
static double YOut[4];


// 64-bit Mersenner Twister *******************************************

/* Period parameters */
#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

/* The array for the state vector */
static unsigned long long mt[NN];
/* mti==NN+1 means mt[NN] is not initialized */
static int mti = NN + 1;

/* initializes mt[NN] with a seed */
void init_genrand64(unsigned long long seed)
{
	mt[0] = seed;
	for (mti = 1; mti<NN; mti++)
		mt[mti] = (6364136223846793005ULL * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
}


/* generates a random number on [0, 2^64-1]-interval */
__inline unsigned long long myrand64(void)
{
	int i;
	unsigned long long x;
	static unsigned long long mag01[2] = { 0ULL, MATRIX_A };

	if (mti >= NN) { /* generate NN words at one time */
		for (i = 0; i<NN - MM; i++) {
			x = (mt[i] & UM) | (mt[i + 1] & LM);
			mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];
		}
		for (; i<NN - 1; i++) {
			x = (mt[i] & UM) | (mt[i + 1] & LM);
			mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];
		}
		x = (mt[NN - 1] & UM) | (mt[0] & LM);
		mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];
		mti = 0;
	}

	x = mt[mti++];
	x ^= (x >> 29) & 0x5555555555555555ULL;
	x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
	x ^= (x << 37) & 0xFFF7EEE000000000ULL;
	x ^= (x >> 43);

	return x;
}

/* generates a random number on (0,1)-real-interval */
__inline double myrandd(void)
{
	return ((myrand64() >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}

// *******************************************

void InitRand(int RS)
{
	uint NI;
	if (RS == 0)
		NI = (uint)time(NULL); //initialization of a random number generator
	else
		NI = RS;
	init_genrand64(NI);
}


double* RunGillespie(double TIni, double TMax, double Y01, double Y02)
{
	double y1, y2, R1;
	double RateSum, Time;
	uint N1, N2;
	
	N1 = Y01 * N;
	N2 = Y02 * N;
    Time = TIni;
	do
	{
			y1 = N1 * CN;
			y2 = N2 * CN;
			R1 = K1 * y1 * (1.0 - y1 - y2);
			RateSum = R1 + K2 * y1;
			if (R1 >= myrandd() * RateSum)
				N1++;
			else 
			{
				N1--; N2++;
			}
			Time -= log(myrandd()) * CN / RateSum;

		} while ((Time < TMax) && (N1 > 0));

	YOut[0] = (N - N1 - N2) * CN;
	YOut[1] = N1 * CN;
	YOut[2] = N2 * CN;
	YOut[3] = Time;

	return YOut;

}


int InitGillespie(int RandS, int SisSize, double ParK1, double ParK2)
{
	K1 = ParK1;
	K2 = ParK2;
	K1 = K1 * 4.0;
	InitRand(RandS);
	N = SisSize;
	CN = 1.0 / N;
	//printf("Init Gillespie: N = %u;  Ki = %.4e %.4e\n", N, K1, K2);
	return 0;
}
