/*
 * Device math mirroring common.h, quaternion.cpp, tree.h and curl.h.
 * Expressions keep the CPU operand order so that the double-precision
 * instantiation (compiled without FMA contraction) tracks the CPU closely.
 */

#ifndef SMINA_GPU_MATH_CUH
#define SMINA_GPU_MATH_CUH

#include <cfloat>
#include "gpu_defs.h"
#include "portable_math.h"

namespace gpusmina
{

GPU_HD float gsqrt(float x) { return sqrtf(x); }
GPU_HD double gsqrt(double x) { return sqrt(x); }
// double precision is the validation build: use the same transcendental
// functions as the CPU validation build (portable_math.h)
GPU_HD float gsin(float x) { return sinf(x); }
GPU_HD double gsin(double x) { return portable_math::sin(x); }
GPU_HD float gcos(float x) { return cosf(x); }
GPU_HD double gcos(double x) { return portable_math::cos(x); }
GPU_HD float gacos(float x) { return acosf(x); }
GPU_HD double gacos(double x) { return portable_math::acos(x); }
GPU_HD float gceil(float x) { return ceilf(x); }
GPU_HD double gceil(double x) { return ceil(x); }
GPU_HD float gfloor(float x) { return floorf(x); }
GPU_HD double gfloor(double x) { return floor(x); }
GPU_HD float gabs(float x) { return fabsf(x); }
GPU_HD double gabs(double x) { return fabs(x); }

// common.h constants (fl is double on the CPU)
template<typename T> GPU_HD T c_pi() { return T(3.1415926535897931); }
template<typename T> GPU_HD T c_epsilon_fl() { return T(DBL_EPSILON); }

template<typename T>
struct vec3
{
	T d[3];
	GPU_HD T& operator[](int i) { return d[i]; }
	GPU_HD const T& operator[](int i) const { return d[i]; }
};

template<typename T> GPU_HD vec3<T> mk(T x, T y, T z)
{
	vec3<T> r;
	r.d[0] = x;
	r.d[1] = y;
	r.d[2] = z;
	return r;
}
template<typename T> GPU_HD vec3<T> load3(const T* p) { return mk(p[0], p[1], p[2]); }
template<typename T> GPU_HD void store3(T* p, const vec3<T>& v)
{
	p[0] = v.d[0];
	p[1] = v.d[1];
	p[2] = v.d[2];
}
template<typename T> GPU_HD vec3<T> operator+(const vec3<T>& a, const vec3<T>& b)
{
	return mk(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}
template<typename T> GPU_HD vec3<T> operator-(const vec3<T>& a, const vec3<T>& b)
{
	return mk(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
template<typename T> GPU_HD vec3<T> operator*(T s, const vec3<T>& v) // s * v
{
	return mk(s * v[0], s * v[1], s * v[2]);
}
template<typename T> GPU_HD void add_to(vec3<T>& a, const vec3<T>& b) // a += b
{
	a.d[0] += b[0];
	a.d[1] += b[1];
	a.d[2] += b[2];
}
template<typename T> GPU_HD void sub_from(vec3<T>& a, const vec3<T>& b) // a -= b
{
	a.d[0] -= b[0];
	a.d[1] -= b[1];
	a.d[2] -= b[2];
}
template<typename T> GPU_HD void scale(vec3<T>& a, T s) // a *= s
{
	a.d[0] *= s;
	a.d[1] *= s;
	a.d[2] *= s;
}
template<typename T> GPU_HD T dot(const vec3<T>& a, const vec3<T>& b) // vec::operator*
{
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
template<typename T> GPU_HD T sqr(T x) { return x * x; }
template<typename T> GPU_HD T norm_sqr(const vec3<T>& v) // sqr(vec)
{
	return sqr(v[0]) + sqr(v[1]) + sqr(v[2]);
}
template<typename T> GPU_HD vec3<T> cross(const vec3<T>& a, const vec3<T>& b)
{
	return mk(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
			a[0] * b[1] - a[1] * b[0]);
}

// column-major 3x3, as mat in common.h
template<typename T>
struct mat3
{
	T d[9];
};
template<typename T> GPU_HD vec3<T> mul(const mat3<T>& m, const vec3<T>& v)
{
	return mk(m.d[0] * v[0] + m.d[3] * v[1] + m.d[6] * v[2],
			m.d[1] * v[0] + m.d[4] * v[1] + m.d[7] * v[2],
			m.d[2] * v[0] + m.d[5] * v[1] + m.d[8] * v[2]);
}

// boost::math::quaternion<fl> (R_component_1..4 = a, b, c, d)
template<typename T>
struct quat
{
	T a, b, c, d;
};
template<typename T> GPU_HD quat<T> mkq(T a, T b, T c, T d)
{
	quat<T> q;
	q.a = a;
	q.b = b;
	q.c = c;
	q.d = d;
	return q;
}
template<typename T> GPU_HD quat<T> qt_identity() { return mkq(T(1), T(0), T(0), T(0)); }
// quaternion::operator*= (Hamilton product, boost expression order)
template<typename T> GPU_HD quat<T> qmul(const quat<T>& l, const quat<T>& r)
{
	return mkq(l.a * r.a - l.b * r.b - l.c * r.c - l.d * r.d,
			l.a * r.b + l.b * r.a + l.c * r.d - l.d * r.c,
			l.a * r.c - l.b * r.d + l.c * r.a + l.d * r.b,
			l.a * r.d + l.b * r.c - l.c * r.b + l.d * r.a);
}
template<typename T> GPU_HD T quaternion_norm_sqr(const quat<T>& q)
{
	return sqr(q.a) + sqr(q.b) + sqr(q.c) + sqr(q.d);
}

template<typename T> GPU_HD void normalize_angle(T& x) // common.h
{
	const T pi = c_pi<T>();
	for (;;)
	{
		if (x > 3 * pi)
		{
			T n = (x - pi) / (2 * pi);
			x -= 2 * pi * gceil(n);
		}
		else if (x < -3 * pi)
		{
			T n = (-x - pi) / (2 * pi);
			x += 2 * pi * gceil(n);
		}
		else
		{
			if (x > pi)
				x -= 2 * pi;
			else if (x < -pi)
				x += 2 * pi;
			return;
		}
	}
}
template<typename T> GPU_HD T normalized_angle(T x)
{
	normalize_angle(x);
	return x;
}

template<typename T> GPU_HD quat<T> angle_to_quaternion(const vec3<T>& axis, T angle)
{
	normalize_angle(angle);
	T c = gcos(angle / 2);
	T s = gsin(angle / 2);
	return mkq(c, s * axis[0], s * axis[1], s * axis[2]);
}

template<typename T> GPU_HD quat<T> angle_to_quaternion(const vec3<T>& rotation)
{
	T angle = gsqrt(norm_sqr(rotation));
	if (angle > c_epsilon_fl<T>())
	{
		vec3<T> axis = (1 / angle) * rotation;
		return angle_to_quaternion(axis, angle);
	}
	return qt_identity<T>();
}

template<typename T> GPU_HD void quaternion_normalize_approx(quat<T>& q)
{
	const T tolerance = T(1e-6);
	const T s = quaternion_norm_sqr(q);
	if (gabs(s - 1) < tolerance)
		; // most likely scenario
	else
	{
		const T a = gsqrt(s);
		const T inv = 1 / a;
		q.a *= inv;
		q.b *= inv;
		q.c *= inv;
		q.d *= inv;
	}
}

template<typename T> GPU_HD void quaternion_increment(quat<T>& q, const vec3<T>& rotation)
{
	q = qmul(angle_to_quaternion(rotation), q);
	quaternion_normalize_approx(q);
}

template<typename T> GPU_HD mat3<T> quaternion_to_r3(const quat<T>& q)
{
	const T a = q.a, b = q.b, c = q.c, d = q.d;
	const T aa = a * a, ab = a * b, ac = a * c, ad = a * d;
	const T bb = b * b, bc = b * c, bd = b * d;
	const T cc = c * c, cd = c * d, dd = d * d;
	mat3<T> m; // m.d[i + 3*j] = tmp(i, j)
	m.d[0] = (aa + bb - cc - dd);
	m.d[3] = 2 * (-ad + bc);
	m.d[6] = 2 * (ac + bd);
	m.d[1] = 2 * (ad + bc);
	m.d[4] = (aa - bb + cc - dd);
	m.d[7] = 2 * (-ab + cd);
	m.d[2] = 2 * (-ac + bd);
	m.d[5] = 2 * (ab + cd);
	m.d[8] = (aa - bb - cc + dd);
	return m;
}

// curl.h (soft curl); not_max(v) holds for every v smina passes
template<typename T> GPU_HD void curl(T& e, vec3<T>& deriv, T v)
{
	if (e > 0)
	{
		T tmp = (v < c_epsilon_fl<T>()) ? T(0) : (v / (v + e));
		e *= tmp;
		scale(deriv, sqr(tmp));
	}
}
template<typename T> GPU_HD void curl(T& e, T& deriv, T v)
{
	if (e > 0)
	{
		T tmp = (v < c_epsilon_fl<T>()) ? T(0) : (v / (v + e));
		e *= tmp;
		deriv *= sqr(tmp);
	}
}
template<typename T> GPU_HD void curl(T& e, T v)
{
	if (e > 0)
	{
		T tmp = (v < c_epsilon_fl<T>()) ? T(0) : (v / (v + e));
		e *= tmp;
	}
}

} // namespace gpusmina

#endif
