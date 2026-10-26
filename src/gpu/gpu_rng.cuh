/*
 * boost::mt19937 and the boost distributions smina draws from during the
 * Monte Carlo search (random.cpp), reproduced bit for bit so a chain's random
 * stream is identical to the CPU's for the same seed.
 *
 * Chains are seeded and randomized on the host with boost::mt19937 exactly as
 * parallel_mc / monte_carlo do; the generator state is then handed over via
 * boost's operator<< (which canonicalizes it so that i == n).
 */

#ifndef SMINA_GPU_RNG_CUH
#define SMINA_GPU_RNG_CUH

#include "gpu_defs.h"

namespace gpusmina
{

const int MT_N = 624;
const int MT_M = 397;

struct MTState
{
	unsigned x[MT_N];
	int i;
};

GPU_HD void mt_twist(MTState& s)
{
	const unsigned upper_mask = (~0u) << 31;
	const unsigned lower_mask = ~upper_mask;
	const unsigned a = 0x9908b0dfu;
	for (int j = 0; j < MT_N - MT_M; j++)
	{
		unsigned y = (s.x[j] & upper_mask) | (s.x[j + 1] & lower_mask);
		s.x[j] = s.x[j + MT_M] ^ (y >> 1) ^ ((s.x[j + 1] & 1) * a);
	}
	for (int j = MT_N - MT_M; j < MT_N - 1; j++)
	{
		unsigned y = (s.x[j] & upper_mask) | (s.x[j + 1] & lower_mask);
		s.x[j] = s.x[j - (MT_N - MT_M)] ^ (y >> 1) ^ ((s.x[j + 1] & 1) * a);
	}
	unsigned y = (s.x[MT_N - 1] & upper_mask) | (s.x[0] & lower_mask);
	s.x[MT_N - 1] = s.x[MT_M - 1] ^ (y >> 1) ^ ((s.x[0] & 1) * a);
	s.i = 0;
}

GPU_HD unsigned mt_next(MTState& s)
{
	if (s.i == MT_N)
		mt_twist(s);
	unsigned z = s.x[s.i];
	++s.i;
	z ^= ((z >> 11) & 0xffffffffu);
	z ^= ((z << 7) & 0x9d2c5680u);
	z ^= ((z << 15) & 0xefc60000u);
	z ^= (z >> 18);
	return z;
}

// random_fl: boost::uniform_real<double> (generate_uniform_real, integral
// engine); always in double like the CPU's fl
GPU_HD double random_fl(double a, double b, MTState& s)
{
	for (;;)
	{
		double numerator = double(mt_next(s) - 0u);
		double divisor = double(0xffffffffu - 0u) + 1;
		double result = numerator / divisor * (b - a) + a;
		if (result < b)
			return result;
	}
}

// random_int: boost::uniform_int<int> (generate_uniform_int with
// brange = 2^32 - 1 > range); valid for a <= b with b - a < 2^32 - 1
GPU_HD int random_int(int a, int b, MTState& s)
{
	const unsigned range = unsigned(b) - unsigned(a);
	if (range == 0)
		return a;
	const unsigned brange = 0xffffffffu;
	unsigned bucket_size = brange / (range + 1);
	if (brange % (range + 1) == range)
		++bucket_size;
	for (;;)
	{
		unsigned result = mt_next(s) / bucket_size;
		if (result <= range)
			return int(result + unsigned(a));
	}
}

} // namespace gpusmina

#endif
