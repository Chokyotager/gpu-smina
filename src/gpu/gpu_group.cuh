/*
 * The set of threads that cooperates on one conformation / Monte Carlo
 * chain.  TPC (threads per chain) is 32 for throughput (one warp per chain,
 * many chains per block) or a multiple of 32 up to 256 for latency (several
 * warps per chain, synchronized with a named barrier).
 *
 * Reductions are deterministic: every thread of the group obtains the same
 * value, combined in a fixed order.
 */

#ifndef SMINA_GPU_GROUP_CUH
#define SMINA_GPU_GROUP_CUH

namespace gpusmina
{

const unsigned FULL_MASK = 0xffffffffu;

// per-group shared scratch used by multi-warp reductions (doubles fit)
const int GROUP_SCRATCH_BYTES = 128; // 8 values (double) + 8 indices

template<int TPC>
struct Group
{
	int rank; // 0 .. TPC-1
	int gid; // group index within the block
	double* red; // GROUP_SCRATCH_BYTES of shared memory (TPC > 32 only)

	__device__ __forceinline__ void sync() const
	{
		if (TPC <= 32)
			__syncwarp();
		else
			asm volatile("bar.sync %0, %1;" :: "r"(gid + 1), "r"(TPC) : "memory");
	}

	template<typename T> __device__ __forceinline__ T sum(T v) const
	{
		for (int o = 16; o > 0; o >>= 1)
			v += __shfl_xor_sync(FULL_MASK, v, o);
		if (TPC <= 32)
			return v;
		T* r = reinterpret_cast<T*>(red);
		sync();
		if ((rank & 31) == 0)
			r[rank >> 5] = v;
		sync();
		T total = r[0];
		for (int w = 1; w < TPC / 32; w++)
			total += r[w];
		sync();
		return total;
	}

	template<typename T> __device__ __forceinline__ T max(T v) const
	{
		for (int o = 16; o > 0; o >>= 1)
			v = fmax(v, __shfl_xor_sync(FULL_MASK, v, o));
		if (TPC <= 32)
			return v;
		T* r = reinterpret_cast<T*>(red);
		sync();
		if ((rank & 31) == 0)
			r[rank >> 5] = v;
		sync();
		T m = r[0];
		for (int w = 1; w < TPC / 32; w++)
			m = fmax(m, r[w]);
		sync();
		return m;
	}

	// value of rank 0
	template<typename T> __device__ __forceinline__ T bcast(T v) const
	{
		if (TPC <= 32)
			return __shfl_sync(FULL_MASK, v, 0);
		T* r = reinterpret_cast<T*>(red);
		sync();
		if (rank == 0)
			r[0] = v;
		sync();
		T out = r[0];
		sync();
		return out;
	}

	// (value, index) minimum, lowest index on ties
	template<typename T> __device__ __forceinline__ void argmin(T& best,
			int& besti) const
	{
		for (int o = 16; o > 0; o >>= 1)
		{
			T ob = __shfl_xor_sync(FULL_MASK, best, o);
			int oi = __shfl_xor_sync(FULL_MASK, besti, o);
			if (ob < best || (ob == best && oi < besti))
			{
				best = ob;
				besti = oi;
			}
		}
		if (TPC <= 32)
			return;
		T* r = reinterpret_cast<T*>(red);
		int* ri = reinterpret_cast<int*>(red + 8);
		sync();
		if ((rank & 31) == 0)
		{
			r[rank >> 5] = best;
			ri[rank >> 5] = besti;
		}
		sync();
		best = r[0];
		besti = ri[0];
		for (int w = 1; w < TPC / 32; w++)
			if (r[w] < best || (r[w] == best && ri[w] < besti))
			{
				best = r[w];
				besti = ri[w];
			}
		sync();
	}
};

} // namespace gpusmina

#endif
