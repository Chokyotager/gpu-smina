/*
 * Deterministic double precision sin, cos, acos and exp that give
 * bit-identical results on the CPU and on CUDA devices, provided neither side
 * contracts multiply-adds into FMAs.  The algorithms and coefficients are
 * those of fdlibm: k_sin.c, k_cos.c and the medium-range reduction of
 * e_rem_pio2.c (sin, cos), e_acos.c (acos) and e_exp.c (exp).
 *
 * Used by the double precision GPU kernels (validation and GPU local
 * optimization) and, when SMINA_PORTABLE_MATH is defined, by the CPU code on
 * the search path, so that CPU and GPU trajectories can be compared bit for
 * bit.  CPU smina otherwise uses the platform libm.  Accuracy is within about
 * 1 ulp.
 *
 * The notices of the fdlibm sources these functions derive from:
 *
 * k_sin.c, k_cos.c, e_rem_pio2.c, e_acos.c:
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 * e_exp.c:
 * ====================================================
 * Copyright (C) 2004 by Sun Microsystems, Inc. All rights reserved.
 *
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#ifndef SMINA_PORTABLE_MATH_H
#define SMINA_PORTABLE_MATH_H

#include <cmath>
#include <cstring>

#ifdef __CUDACC__
#define PM_HD __host__ __device__ inline
#else
#define PM_HD inline
#endif

namespace portable_math
{

PM_HD long long bits(double x)
{
#ifdef __CUDA_ARCH__
	return __double_as_longlong(x);
#else
	long long b;
	std::memcpy(&b, &x, sizeof(b));
	return b;
#endif
}

PM_HD double from_bits(long long b)
{
#ifdef __CUDA_ARCH__
	return __longlong_as_double(b);
#else
	double x;
	std::memcpy(&x, &b, sizeof(x));
	return x;
#endif
}

// x * 2^k for |k| within the normal exponent range of the result
PM_HD double scale2(double x, int k)
{
	if (k > 1023)
	{
		x *= from_bits((long long) (1023 + 1023) << 52);
		k -= 1023;
	}
	else if (k < -1022)
	{
		x *= from_bits((long long) (-1000 + 1023) << 52);
		k += 1000;
		if (k < -1022)
		{
			x *= from_bits((long long) (-1000 + 1023) << 52);
			k += 1000;
		}
	}
	return x * from_bits((long long) (k + 1023) << 52);
}

// round to nearest integer (ties to even), exact for |x| < 2^51
PM_HD double rint_d(double x)
{
	const double big = 6755399441055744.0; // 1.5 * 2^52
	if (x >= 0)
		return (x + big) - big;
	return -((-x + big) - big);
}

// sin and cos on [-pi/4, pi/4]
PM_HD double ksin(double x)
{
	const double S1 = -1.66666666666666324348e-01, S2 = 8.33333333332248946124e-03,
			S3 = -1.98412698298579493134e-04, S4 = 2.75573137070700676789e-06,
			S5 = -2.50507602534068634195e-08, S6 = 1.58969099521155010221e-10;
	const double z = x * x;
	const double v = z * x;
	const double r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
	return x + v * (S1 + z * r);
}

PM_HD double kcos(double x)
{
	const double C1 = 4.16666666666666019037e-02, C2 = -1.38888888888741095749e-03,
			C3 = 2.48015872894767294178e-05, C4 = -2.75573143513906633035e-07,
			C5 = 2.08757232129817482790e-09, C6 = -1.13596475577881948265e-11;
	const double z = x * x;
	const double r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
	if (fabs(x) < 0.3)
		return 1.0 - (0.5 * z - z * r);
	// fdlibm: subtract a dyadic qx so that 1 - qx is exact
	const double qx = (fabs(x) > 0.78125) ? 0.28125 : from_bits(
			bits(z * 0.25) & ~0xffffffffLL);
	const double hz = 0.5 * z - qx;
	const double a = 1.0 - qx;
	return a - (hz - z * r);
}

// x = k * pi/2 + r, |r| <= ~pi/4 (Cody-Waite, valid for moderate |x|)
PM_HD double reduce(double x, int& k)
{
	const double invpio2 = 6.36619772367581382433e-01;
	const double pio2_1 = 1.57079632673412561417e+00; // first 33 bits of pi/2
	const double pio2_1t = 6.07710050650619224932e-11; // pi/2 - pio2_1
	const double kd = rint_d(x * invpio2);
	k = int(kd);
	return (x - kd * pio2_1) - kd * pio2_1t;
}

PM_HD double sin(double x)
{
	if (fabs(x) <= 7.85398163397448278999e-01)
		return ksin(x);
	int k;
	const double r = reduce(x, k);
	switch (k & 3)
	{
	case 0:
		return ksin(r);
	case 1:
		return kcos(r);
	case 2:
		return -ksin(r);
	default:
		return -kcos(r);
	}
}

PM_HD double cos(double x)
{
	if (fabs(x) <= 7.85398163397448278999e-01)
		return kcos(x);
	int k;
	const double r = reduce(x, k);
	switch (k & 3)
	{
	case 0:
		return kcos(r);
	case 1:
		return -ksin(r);
	case 2:
		return -kcos(r);
	default:
		return ksin(r);
	}
}

PM_HD double acos(double x)
{
	const double pi = 3.14159265358979311600e+00;
	const double pio2_hi = 1.57079632679489655800e+00;
	const double pio2_lo = 6.12323399573676603587e-17;
	const double pS0 = 1.66666666666666657415e-01, pS1 = -3.25565818622400915405e-01,
			pS2 = 2.01212532134862925881e-01, pS3 = -4.00555345006794114027e-02,
			pS4 = 7.91534994289814532176e-04, pS5 = 3.47933107596021167570e-05;
	const double qS1 = -2.40339491173441421878e+00, qS2 = 2.02094576023350569471e+00,
			qS3 = -6.88283971605453293030e-01, qS4 = 7.70381505559019352791e-02;
	const double ax = fabs(x);
	if (ax >= 1)
	{
		if (x == 1)
			return 0;
		if (x == -1)
			return pi;
		return (x - x) / (x - x); // NaN
	}
	if (ax < 0.5)
	{
		const double z = x * x;
		const double p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4
				+ z * pS5)))));
		const double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
		const double r = p / q;
		return pio2_hi - (x - (pio2_lo - x * r));
	}
	if (x < 0)
	{
		const double z = (1.0 + x) * 0.5;
		const double p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4
				+ z * pS5)))));
		const double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
		const double s = sqrt(z);
		const double r = p / q;
		const double w = r * s - pio2_lo;
		return pi - 2.0 * (s + w);
	}
	const double z = (1.0 - x) * 0.5;
	const double s = sqrt(z);
	const double df = from_bits(bits(s) & ~0xffffffffLL);
	const double c = (z - df * df) / (s + df);
	const double p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4
			+ z * pS5)))));
	const double q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
	const double r = p / q;
	const double w = r * s + c;
	return 2.0 * (df + w);
}

PM_HD double exp(double x)
{
	const double ln2_hi = 6.93147180369123816490e-01;
	const double ln2_lo = 1.90821492927058770002e-10;
	const double invln2 = 1.44269504088896338700e+00;
	const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03,
			P3 = 6.61375632143793436117e-05, P4 = -1.65339022054652515390e-06,
			P5 = 4.13813679705723846039e-08;
	if (x != x)
		return x;
	if (x > 7.09782712893383973096e+02)
		return from_bits(0x7ff0000000000000LL); // +inf
	if (x < -7.45133219101941108420e+02)
		return 0;
	double hi = x, lo = 0;
	int k = 0;
	const double ax = fabs(x);
	if (ax > 0.5 * 6.93147180559945286227e-01)
	{
		k = int(invln2 * x + ((x < 0) ? -0.5 : 0.5));
		hi = x - k * ln2_hi;
		lo = k * ln2_lo;
		x = hi - lo;
	}
	else if (ax < 3.7252902984e-09) // 2^-28
		return 1.0 + x;
	const double t = x * x;
	const double c = x - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
	if (k == 0)
		return 1.0 - ((x * c) / (c - 2.0) - x);
	const double y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi);
	return scale2(y, k);
}

} // namespace portable_math

#endif
