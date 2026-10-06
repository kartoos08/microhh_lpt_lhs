#include <cuda_runtime.h>
#include <math.h>

#include "lpt_interp6.h"
#include "lpt_microphysics.h"

namespace Lpt
{

template<typename TF>
__device__ inline TF wrap_periodic(TF x, const TF length)
{
    x -= floor(x/length)*length;
    return (x >= length) ? x-length : ((x < TF(0)) ? x+length : x);
}

template<typename TF>
__device__ inline void atomic_add_tf(TF* p, TF v)
{
    atomicAdd(p, v);
}

// Deposit the supersaturation tendency with positive CIC weights. The Lagrange
// interpolant is intentionally NOT reused for deposition because its negative
// lobes can create nonphysical local vapor sources.
template<typename TF>
__device__ inline void deposit_supersaturation_cic(
        TF* const st,
        const TF x, const TF y, const TF z,
        const TF dSdt,
        const int nx, const int ny, const int nz,
        const int istart, const int jstart, const int kstart,
        const int jstride, const int kstride,
        const TF dx, const TF dy, const TF dz)
{
    const TF gx = x/dx - TF(0.5);
    const TF gy = y/dy - TF(0.5);
    const TF gz = z/dz - TF(0.5);
    const int i0 = int(floor(gx));
    const int j0 = int(floor(gy));
    const int k0 = int(floor(gz));
    const TF fx = gx - TF(i0);
    const TF fy = gy - TF(j0);
    const TF fz = gz - TF(k0);

    for (int dk=0; dk<2; ++dk)
        for (int dj=0; dj<2; ++dj)
            for (int di=0; di<2; ++di)
            {
                const TF wx = di ? fx : TF(1)-fx;
                const TF wy = dj ? fy : TF(1)-fy;
                const TF wz = dk ? fz : TF(1)-fz;
                const int i = istart + imod(i0+di, nx);
                const int j = jstart + imod(j0+dj, ny);
                const int k = kstart + imod(k0+dk, nz);
                atomic_add_tf(st + i + j*jstride + k*kstride, wx*wy*wz*dSdt);
            }
}

template<typename TF>
__global__ void advance_droplets_g(
        TF* const __restrict__ x,
        TF* const __restrict__ y,
        TF* const __restrict__ z,
        TF* const __restrict__ up,
        TF* const __restrict__ vp,
        TF* const __restrict__ wp,
        TF* const __restrict__ r2,
        const TF* const __restrict__ r_ccn,
        TF* const __restrict__ chi_local,
        TF* const __restrict__ s_tendency,
        const Periodic_field_view<TF> uview,
        const Periodic_field_view<TF> vview,
        const Periodic_field_view<TF> wview,
        const Periodic_field_view<TF> sview,
        const Growth_config<TF> cfg,
        const TF dt,
        const TF lx, const TF ly, const TF lz,
        const std::size_t np,
        const bool two_way)
{
    const std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (p >= np)
        return;

    TF S, dSdx, dSdy, dSdz;
    sample6_value_gradient(sview, x[p], y[p], z[p], S, dSdx, dSdy, dSdz);
    chi_local[p] = TF(2)*cfg.kappa_s*(dSdx*dSdx + dSdy*dSdy + dSdz*dSdz);

    const TF uf = sample6(uview, x[p], y[p], z[p]);
    const TF vf = sample6(vview, x[p], y[p], z[p]);
    const TF wf = sample6(wview, x[p], y[p], z[p]);

    const TF r2_floor = r_ccn[p]*r_ccn[p];
    const TF old_r2 = r2[p];
    const TF trial = old_r2 + TF(2)*cfg.G*S*dt;
    const TF new_r2 = (trial < r2_floor) ? r2_floor : trial;
    r2[p] = new_r2;

    const TF old_r = sqrt((old_r2 > r2_floor) ? old_r2 : r2_floor);
    const TF new_r = sqrt(new_r2);
    const TF dm_liquid = TF(4.1887902047863909846)*cfg.rho_w*
            (new_r*new_r*new_r - old_r*old_r*old_r);

    if (two_way && s_tendency != nullptr && dm_liquid != TF(0))
    {
        const TF cell_volume = sview.dx*sview.dy*sview.dz;
        // qv source = -dm_liquid * particle_weight / (rho_air * V * dt)
        // S = qv/qvs_ref - 1 for the isothermal Boussinesq scalar used here.
        const TF dSdt = -dm_liquid*cfg.particle_weight /
                (cfg.rho_air*cell_volume*cfg.qvs_ref*dt);
        deposit_supersaturation_cic(
                s_tendency, x[p], y[p], z[p], dSdt,
                sview.nx, sview.ny, sview.nz,
                sview.istart, sview.jstart, sview.kstart,
                sview.jstride, sview.kstride,
                sview.dx, sview.dy, sview.dz);
    }

    // Stokes response, exact for frozen fluid velocity over this split step.
    const TF tau_p = TF(2)*cfg.rho_w*new_r2 / (TF(9)*cfg.rho_air*cfg.nu_air);
    const TF e = exp(-dt/tau_p);
    up[p] = uf + (up[p]-uf)*e;
    vp[p] = vf + (vp[p]-vf)*e;
    const TF w_eq = wf - cfg.gravity*tau_p;
    wp[p] = w_eq + (wp[p]-w_eq)*e;

    x[p] = wrap_periodic(x[p] + dt*up[p], lx);
    y[p] = wrap_periodic(y[p] + dt*vp[p], ly);
    z[p] = wrap_periodic(z[p] + dt*wp[p], lz);
}

// Call this wrapper from Lpt::Droplets<TF>::exec() in your fork.
template<typename TF>
void launch_advance_droplets(
        TF* x, TF* y, TF* z,
        TF* up, TF* vp, TF* wp,
        TF* r2, const TF* r_ccn, TF* chi_local,
        TF* s_tendency,
        Periodic_field_view<TF> uview,
        Periodic_field_view<TF> vview,
        Periodic_field_view<TF> wview,
        Periodic_field_view<TF> sview,
        Growth_config<TF> cfg,
        TF dt, TF lx, TF ly, TF lz,
        std::size_t np, bool two_way)
{
    constexpr int block = 256;
    const int grid = int((np + block - 1)/block);
    advance_droplets_g<TF><<<grid, block>>>(
            x,y,z,up,vp,wp,r2,r_ccn,chi_local,s_tendency,
            uview,vview,wview,sview,cfg,dt,lx,ly,lz,np,two_way);
}

#ifdef FLOAT_SINGLE
template void launch_advance_droplets<float>(
        float*,float*,float*,float*,float*,float*,float*,const float*,float*,float*,
        Periodic_field_view<float>,Periodic_field_view<float>,Periodic_field_view<float>,Periodic_field_view<float>,
        Growth_config<float>,float,float,float,float,std::size_t,bool);
#else
template void launch_advance_droplets<double>(
        double*,double*,double*,double*,double*,double*,double*,const double*,double*,double*,
        Periodic_field_view<double>,Periodic_field_view<double>,Periodic_field_view<double>,Periodic_field_view<double>,
        Growth_config<double>,double,double,double,double,std::size_t,bool);
#endif

} // namespace Lpt
