
// SIRS on a square lattice
// KMC with real time

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "KMCDLL1.h"

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned char byte;

#define  NNeigh1 4
#define  MaxNRate 24

static uint N1, N2, N1xN2, N1xN2_1;
static double DN1xN2, Time;
static char *Surf, *NSur1, *NSur2;

struct NeiInfoType
{
	uint Nei1, Nei2, Nei3, Nei4;
};
struct NeiInfoType *Neis;


static uint NAds, NAds1, NAds2;
static int NRates, SaveSurfStep, SSStep = 0, FirstRateSp1, ErrCode = 0, Version = 0, NRec = 0;

static char RateInfo1[5], RateInfo2[5], N0ForRate[MaxNRate];

static double CurRate[MaxNRate], DifRate[MaxNRate], SiteRate[MaxNRate];
static double K2Rate[MaxNRate], K1Rate[MaxNRate], RelDifRate[MaxNRate], RelK2Rate[MaxNRate];

struct RateStruc
{
  int NTot;
  int *RateI;
};

struct SiteInfoStruc
{
  char NRate;
  int Key;
};

struct RateStruc Rates[MaxNRate];
struct SiteInfoStruc *SiteInfo;

static double DeltaTime1, TimeStopCalc, PrevTimeOut, K1, K2, K3, KDif1, KDif2;

static double YOut[4];

// 64-bit Mersenner Twister *******************************************

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL 
#define LM 0x7FFFFFFFULL 

static unsigned long long mt[NN];
static int mti = NN + 1;

//initializes mt[NN] with a seed 
void init_genrand64(unsigned long long seed)
{
	mt[0] = seed;
	for (mti = 1; mti<NN; mti++)
		mt[mti] = (6364136223846793005ULL * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
}


// generates a random number on [0, 2^64-1]-interval 
__inline unsigned long long myrand64(void)
{
	int i;
	unsigned long long x;
	static unsigned long long mag01[2] = { 0ULL, MATRIX_A };

	if (mti >= NN) { 
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

// generates a random number on (0,1)-real-interval 
__inline double myrandd(void)
{
	return ((myrand64() >> 12) + 0.5) * (1.0 / 4503599627370496.0);
}
//======================================================================
// end of Mersenner Twister 

void InitRand(int RS)
  {
	uint NI;
    if (RS == 0)
	  NI = (uint)time(NULL); //initialization of a random number generator
	else
	  NI = RS;
	init_genrand64(NI);
	//printf("RN_Init done, RC=%u\n", NI);
  }



  void Set2DGrid1(void) // "by rows" enumeration of sites; periodic b.c.; square lattice
  {
    int I, K, J, IM;
  	for (I = 1; I <= N1xN2; I++)
	{
	  Neis[I].Nei2 = I + 1;
      Neis[I].Nei4 = I - 1;
  	  J = I + N2;
      if (J > N1xN2)
	  {
        J -= N1xN2;
	  }
      Neis[I].Nei1 = J;
  	  J = I - N2;
      if (J <= 0)
	  {
        J += N1xN2;
	  }
      Neis[I].Nei3 = J;
    }
	IM = 0;
    for (I = 1; I <= N1; I++)
	{
      K = IM + 1;
      IM += N2;
      Neis[K].Nei4 = IM;
      Neis[IM].Nei2 = K;
    }

  }



  void SetIniSurf(double Y1_0, double Y2_0)
  {
	uint I, NA, CurNum;
    
	memset(Surf,  0, (N1xN2_1));
	memset(NSur1, 0, (N1xN2_1));
	memset(NSur2, 0, (N1xN2_1));

	for (NA = 1; NA <= 2; NA++)
	{
		if (NA == 1)
	      CurNum = (uint)(Y1_0 * N1xN2 + 0.5);
		else
  		  CurNum = (uint)(Y2_0 * N1xN2 + 0.5);

	    while (CurNum > 0)
		{
	      if (Surf[I = ((myrand64()) % N1xN2) + 1] == 0)
		  {
            CurNum--;
		    Surf[I] = NA;
		  }
		}
	}
  }

 

  void SetSurfInfo(char *Sur)
  {
    int I;
    char *Np;
	char C;
	uint *np; 
	NAds1 = 0;
	NAds2 = 0;
    for (I = 1; I <= N1xN2; I++)
	{
      if ((C = Sur[I]) != 0)
	  {
		if (C == 1)
		{
		  Np = NSur1;
	      NAds1++;
		}
		else
		{
		  Np = NSur2;
	      NAds2++;
		}
        np = &Neis[I].Nei1;
        Np[*np++]++;
        Np[*np++]++;
        Np[*np++]++;
        Np[*np]++;
	  }
	}
  }


 

void InitRate(int NR)
{
	Rates[NR].NTot = 0;
	Rates[NR].RateI = (int *)calloc((N1xN2), sizeof(int));
}

 
  void SetRates(void)
  {
	int n0;

    memset(RateInfo1, 0, sizeof(RateInfo1));
    memset(RateInfo2, 0, sizeof(RateInfo2));
    memset(SiteRate, 0, sizeof(SiteRate));
    memset(RelDifRate, 0, sizeof(RelDifRate));
    memset(RelK2Rate, 0, sizeof(RelK2Rate));
	
	SiteRate[0] = 0;
	NRates = 0;
	for (n0 = NNeigh1; n0 >= 0; n0--)
	  if (KDif2*n0 > 0)
		{
		    NRates++;
		    N0ForRate[NRates] = n0;
		    DifRate[NRates] = (KDif2*n0);
		    RateInfo2[n0] = NRates;
		    SiteRate[NRates] = DifRate[NRates];
		}

	FirstRateSp1 = NRates+1;
	for (n0 = NNeigh1; n0 >= 0; n0--)
	  {
				  NRates++;
				  N0ForRate[NRates] = n0;
				  RateInfo1[n0] = NRates;
				  K1Rate[NRates] = K1*n0;
				  K2Rate[NRates] = K2;
				  DifRate[NRates] = (KDif1*n0);
				  SiteRate[NRates] = DifRate[NRates] + K2Rate[NRates] + K1Rate[NRates];
				  RelDifRate[NRates] = (DifRate[NRates] / SiteRate[NRates]);
				  RelK2Rate[NRates] = RelDifRate[NRates] + (K2Rate[NRates] / SiteRate[NRates]);
	  }
  }

  
  void CalcAllRates(void)
  {
	int I, NR;
	char C;
	struct RateStruc *p;
    struct SiteInfoStruc *s;

	SiteInfo[0].Key = 0;
	SiteInfo[0].NRate = 0;
	for (I = 1; I <= N1xN2; I++)
	{
	  s = &SiteInfo[I];
	  if ((C = Surf[I]) > 0)
	  {
	    if (C == 1)
		  NR = RateInfo1[NNeigh1 - NSur1[I] - NSur2[I]];
		else
		  NR = RateInfo2[NNeigh1 - NSur1[I] - NSur2[I]];
		if (NR > 0)
		{
	      p = &Rates[s->NRate = NR];
	      p->RateI[s->Key =++(p->NTot)] = I;
		}
		else
		{
		  s->NRate = 0;
		  s->Key = 0;
		}
	  }
	  else
	  {
		s->NRate = 0;
		s->Key = 0;
	  }
	}

  }


  int CheckAllRates(void)
  {
	int NR, I, K, J, N1, N2, NRR;
	uint *npi;

	for (NR = 1; NR <= NRates; NR++)
	{
		if (Rates[NR].NTot < 0)
		{
				printf("wrong NTot\n");
				return 1;
		}

		for (K = 1; K <= Rates[NR].NTot; K++) 
		{
			I = Rates[NR].RateI[K];
            if ((SiteInfo[I].Key != K) || (SiteInfo[I].NRate != NR))
			{
				printf("wrong siteinfo\n");
				return 1;
			}
			N1 = 0;
			N2 = 0;
            npi = &Neis[I].Nei1;
		    for (J = 1; J <= 4; J++) 
			{
				if (Surf[*npi] == 1)
					N1++;
				if (Surf[*npi++] == 2)
					N2++;
			}
            if ((N1 != NSur1[I]) || (N2 != NSur2[I]))
			{
				printf("wrong NSur1 or 2\n");
				return 1;
			}

	    if (Surf[I] == 1)
			NRR = RateInfo1[NNeigh1 - NSur1[I] - NSur2[I]];
		else
		  NRR = RateInfo2[NNeigh1 - NSur1[I] - NSur2[I]];

            if (NRR != NR)
			{
				printf("wrong NR %u  %u\n", NR, NRR);
				return 1;
			}
		}
	}

    printf("check - ok!\n");
	
	return 0;

  }

 

__inline void ChangeRate(int I)
{
	int J, NR;
	struct RateStruc *p;
    struct SiteInfoStruc *s;
	s = &SiteInfo[I];
	if (s->Key != 0)
	{
		p = &Rates[s->NRate];
		J = p->RateI[p->NTot--];
		p->RateI[SiteInfo[J].Key = s->Key] = J;
	}
	if (Surf[I] == 1)
		NR = RateInfo1[NNeigh1 - NSur1[I] - NSur2[I]];
	else
	  NR = RateInfo2[NNeigh1 - NSur1[I] - NSur2[I]];
	p = &Rates[s->NRate = NR];
	p->RateI[s->Key = ++(p->NTot)] = I;
}



__inline void ChangeRate0(int I)
{
	int J;
	struct RateStruc *p;
    struct SiteInfoStruc *s;
	s = &SiteInfo[I];
	if (s->Key != 0)
	{
      p = &Rates[s->NRate];
	  J = p->RateI[p->NTot--];
	  p->RateI[SiteInfo[J].Key = s->Key] = J;
	  s->Key = 0;
	}
}



__inline void UpdateRateK3(int I)
  {
	int K;
	uint *npi1, *np1; 

    NAds2--;
	Surf[I] = 0;
	npi1 = np1 = &Neis[I].Nei1;
	for (K = NNeigh1; K--; npi1++)
	  NSur2[*npi1]--;

	ChangeRate0(I);
	for (K = NNeigh1; K--; np1++)
	  if (Surf[*np1] != 0)
		ChangeRate(*np1);

 }

  __inline void UpdateRateK2(int I)
  {
	  int K;
	  uint *npi1, *pi1;

	  NAds1--;
	  NAds2++;
	  Surf[I] = 2;
	  npi1 = pi1 = &Neis[I].Nei1;
	  for (K = NNeigh1; K--; pi1++)
	  {
		  NSur1[*pi1]--;
		  NSur2[*pi1]++;
	  }

	 ChangeRate(I);
	  for (K = NNeigh1; K--; npi1++)
	  if (Surf[*npi1] != 0)
		 ChangeRate(*npi1);

  }
  
  __inline void UpdateRateK1(int I, int dir)
  {
	int K, J;
	uint *np, *npj1,*pj1; 
    
	np = &Neis[I].Nei1;
	for (K = 0;; np++)
	  if ((Surf[*np] == 0) && (K++ == dir))
		break;

	NAds1++;
	J = *np;
	Surf[J] = 1;
	npj1 = pj1 = &Neis[J].Nei1;
	for (K = NNeigh1; K--; npj1++)
	  NSur1[*npj1]++;

	ChangeRate(J);
	for (K = NNeigh1; K--; pj1++)
	  if (Surf[*pj1] != 0)
		ChangeRate(*pj1);

 }

 
  __inline void UpdateDif(int I, int dir)
  {
    int K, J;
	uint *np, *npi1,*npj1,*pi1,*pj1; 
    char *Np1;

    npi1 = np = pi1 = &Neis[I].Nei1;
	for (K = 0;; np++)
	  if ((Surf[*np] == 0) && (K++ == dir))
		break;

	if ((Surf[J = *np] = Surf[I]) == 1)
	  Np1 = NSur1;
	else
	  Np1 = NSur2;
	Surf[I] = 0;
	
	npj1 = pj1 = &Neis[J].Nei1;
    for (K = NNeigh1; K--; pi1++, pj1++)
	{
	  Np1[*pi1]--;
	  Np1[*pj1]++;
	}

	ChangeRate0(I);
	for (K = NNeigh1; K--; npi1++, npj1++)
	{
      if (Surf[*npi1] != 0)
		ChangeRate(*npi1);
	  if (Surf[*npj1] != 0)
		ChangeRate(*npj1);
	}

  }

  double *  RunKMC1(int RunType, double TIni, double TMax, double Y01, double Y02)
  {
	int I, NR;
	uint RN;
	double RN1, RateK3, D, RateProcAll, TimeStep, *p1;


		if (RunType == 0) // new initial conditions
		{
			for (I = 0; I <= NRates; I++)
				Rates[I].NTot = 0;
			SetIniSurf(Y01, Y02);
			SetSurfInfo(Surf);
			CalcAllRates();
			Time = TIni;
		}

		do
		{

			RateK3 = K3*NAds2;
			RateProcAll = RateK3;

			for (NR = 1; NR <= NRates; NR++)
				RateProcAll += (CurRate[NR] = Rates[NR].NTot*SiteRate[NR]);
		
			//TimeStep = 1.0 / RateProcAll;
			TimeStep = -log(myrandd()) / RateProcAll;
			Time += TimeStep;

			D = myrandd() * RateProcAll;
			if ((D -= RateK3) <= 0.0)
			{
				do {} while (Surf[I = (myrand64() % N1xN2 + 1)] != 2);
				UpdateRateK3(I);
			}
			else
			{
				p1 = &CurRate[1];
				for (NR = 1;; NR++)
				if ((D -= *p1++) <= 0.0)
					break;

				RN = myrand64();
				I = Rates[NR].RateI[(RN % Rates[NR].NTot)+1];
		
				if (NR < FirstRateSp1)
					UpdateDif(I, RN % N0ForRate[NR]);
				else
				{
					RN1 = myrandd();
					if (RN1 <= RelDifRate[NR])
						UpdateDif(I, RN % N0ForRate[NR]);
					else
						if (RN1 <= RelK2Rate[NR])
							UpdateRateK2(I);
						else
							UpdateRateK1(I, RN % N0ForRate[NR]);
				}
			}

		}
		while (Time < TMax);

		YOut[0] = (N1xN2 - NAds1 - NAds2) * DN1xN2;
		YOut[1] = NAds1 * DN1xN2;
		YOut[2] = NAds2 * DN1xN2;
		YOut[3] = Time;

		return YOut;

  }

  int InitKMC1(int RandSeed, int LatN1, int LatN2, double ParK1, double ParK2, double ParK3, double ParDif1, double ParDif2)
  {
	  int I, ErrCode = 0;

	  N1 = LatN1;
	  N2 = LatN2;
	  N1xN2 = N1*N2;
	  N1xN2_1 = N1xN2 + 1;
	  DN1xN2 = 1.0 / N1xN2;

	  K1 = ParK1;
	  K2 = ParK2;
	  K3 = ParK3;
	  KDif1 = ParDif1;
	  KDif2 = ParDif2;

	  NAds = 2;
	  Surf = (char *)calloc((N1xN2_1), sizeof(char));
	  NSur1 = (char *)calloc((N1xN2_1), sizeof(char));
	  NSur2 = (char *)calloc((N1xN2_1), sizeof(char));

	  SiteInfo = (void *)calloc((N1xN2_1), 2*(sizeof(int)+sizeof(char)));
	  Neis = (void *)calloc((N1xN2_1), 2*4*sizeof(int));

	  //printf("int size=%u %u\n", sizeof(int), sizeof(char));

	  if (Neis == NULL)
	  {
		  printf("Can't allocate memory\n");
		  ErrCode = 1;
		  return 1;
	  }
	  Set2DGrid1();
	  InitRand(RandSeed);

	  SetRates();
	  for (I = 0; I <= NRates; I++)
	  {
		  InitRate(I);
		  if (Rates[I].RateI == NULL)
		  {
			  printf("Can't allocate memory\n");
			  ErrCode = 1;
			  return 1;
		  }
	  }

	  return ErrCode;

  }

  int FreeMemoryKMC1(void)
  {

	  free(Surf); 
	  free(NSur1);
	  free(NSur2);
	  free(SiteInfo);
	  free(Neis);
	  return 0;

  }

