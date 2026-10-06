#ifndef LPT_MICROPHYSICS_H
#define LPT_MICROPHYSICS_H

#ifdef __CUDACC__
#define LPT_HD __host__ __device__ __forceinline__
#else
#define LPT_HD inline
#endif

template<typename TF>
LPT_HD TF lpt_advance_r2_clamped(const TF r2, const TF r_ccn, const TF S,
                                 const TF G, const TF dt)
{
    const TF floor2 = r_ccn*r_ccn;
    const TF trial = r2 + TF(2)*G*S*dt;
    return trial > floor2 ? trial : floor2;
}

#undef LPT_HD
#endif
