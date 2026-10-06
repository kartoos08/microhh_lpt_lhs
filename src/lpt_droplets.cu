#ifdef USECUDA
#include "lpt_droplets.h"
#include "lpt_microphysics.h"
#include "grid.h"
#include "fields.h"
#include "cuda_buffer.h"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace
{
    __device__ __forceinline__ int imod_g(int i, int n)
    {
        i %= n;
        return i < 0 ? i+n : i;
    }

    template<typename TF>
    __device__ __forceinline__ TF wrap_g(TF x, TF L)
    {
        x -= L*floor(x/L);
        return x >= L ? TF(0) : x;
    }

    template<typename TF>
    __device__ __forceinline__ void w6(const TF x, TF* w)
    {
        w[0] = -x*(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(1))/TF(120);
        w[1] =  x*(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(2))/TF(24);
        w[2] = -(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(12);
        w[3] =  x*(x-TF(3))*(x-TF(2))*(x+TF(1))*(x+TF(2))/TF(12);
        w[4] = -x*(x-TF(3))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(24);
        w[5] =  x*(x-TF(2))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(120);
    }

    template<typename TF>
    __device__ __forceinline__ TF interp6_g(
            const TF* __restrict__ f, TF x, TF y, TF z,
            TF dx, TF dy, TF dz, int nx, int ny, int nz,
            int istart, int jstart, int kstart, int jj, int kk,
            int lx, int ly, int lz)
    {
        const TF qx = x/dx - (lx ? TF(0) : TF(0.5));
        const TF qy = y/dy - (ly ? TF(0) : TF(0.5));
        const TF qz = z/dz - (lz ? TF(0) : TF(0.5));
        const int i0 = int(floor(qx)), j0 = int(floor(qy)), k0 = int(floor(qz));
        TF wx[6], wy[6], wz[6];
        w6(qx-TF(i0), wx); w6(qy-TF(j0), wy); w6(qz-TF(k0), wz);
        TF out = TF(0);
        #pragma unroll
        for (int c=0; c<6; ++c)
        #pragma unroll
        for (int b=0; b<6; ++b)
        #pragma unroll
        for (int a=0; a<6; ++a)
        {
            const int i = istart + imod_g(i0+a-2,nx);
            const int j = jstart + imod_g(j0+b-2,ny);
            const int k = kstart + imod_g(k0+c-2,nz);
            out += wx[a]*wy[b]*wz[c]*f[i+j*jj+k*kk];
        }
        return out;
    }

    template<typename TF>
    __global__ void calc_chi_kernel(
            TF* __restrict__ chi, const TF* __restrict__ s,
            TF kappa_T, TF dx, TF dy, TF dz,
            int nx, int ny, int nz, int istart, int jstart, int kstart, int jj, int kk)
    {
        const int n = nx*ny*nz;
        const int q = blockIdx.x*blockDim.x + threadIdx.x;
        if (q >= n) return;
        const int ii = q % nx;
        const int jj0 = (q/nx) % ny;
        const int kk0 = q/(nx*ny);
        auto val = [&](int di, int dj, int dk) {
            const int i=istart+imod_g(ii+di,nx);
            const int j=jstart+imod_g(jj0+dj,ny);
            const int k=kstart+imod_g(kk0+dk,nz);
            return s[i+j*jj+k*kk];
        };
        const TF sx=(-val(2,0,0)+TF(8)*val(1,0,0)-TF(8)*val(-1,0,0)+val(-2,0,0))/(TF(12)*dx);
        const TF sy=(-val(0,2,0)+TF(8)*val(0,1,0)-TF(8)*val(0,-1,0)+val(0,-2,0))/(TF(12)*dy);
        const TF sz=(-val(0,0,2)+TF(8)*val(0,0,1)-TF(8)*val(0,0,-1)+val(0,0,-2))/(TF(12)*dz);
        const int i=istart+ii, j=jstart+jj0, k=kstart+kk0;
        chi[i+j*jj+k*kk]=TF(2)*kappa_T*(sx*sx+sy*sy+sz*sz);
    }

    template<typename TF>
    __global__ void lpt_kernel(
            std::size_t np,
            TF* __restrict__ xp, TF* __restrict__ yp, TF* __restrict__ zp,
            TF* __restrict__ up, TF* __restrict__ vp, TF* __restrict__ wp,
            TF* __restrict__ r2, const TF* __restrict__ r_ccn, TF* __restrict__ chi_local,
            const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
            const TF* __restrict__ s, const TF* __restrict__ chi,
            TF dt, TF growth_G, TF rho_l, TF mu_air, bool gravity,
            TF Lx, TF Ly, TF Lz, TF dx, TF dy, TF dz,
            int nx, int ny, int nz, int istart, int jstart, int kstart, int jj, int kk)
    {
        const std::size_t p = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
        if (p >= np) return;
        const TF x=xp[p], y=yp[p], z=zp[p];
        const TF uf=interp6_g(u,x,y,z,dx,dy,dz,nx,ny,nz,istart,jstart,kstart,jj,kk,1,0,0);
        const TF vf=interp6_g(v,x,y,z,dx,dy,dz,nx,ny,nz,istart,jstart,kstart,jj,kk,0,1,0);
        const TF wf=interp6_g(w,x,y,z,dx,dy,dz,nx,ny,nz,istart,jstart,kstart,jj,kk,0,0,1);
        const TF Sp=interp6_g(s,x,y,z,dx,dy,dz,nx,ny,nz,istart,jstart,kstart,jj,kk,0,0,0);
        chi_local[p]=interp6_g(chi,x,y,z,dx,dy,dz,nx,ny,nz,istart,jstart,kstart,jj,kk,0,0,0);

        const TF taup=max(TF(2)*rho_l*r2[p]/(TF(9)*mu_air),TF(1.e-12));
        const TF a=exp(-dt/taup);
        TF pu=uf+(up[p]-uf)*a;
        TF pv=vf+(vp[p]-vf)*a;
        TF pw=wf+(wp[p]-wf)*a;
        if (gravity) pw-=TF(9.80665)*taup*(TF(1)-a);
        up[p]=pu; vp[p]=pv; wp[p]=pw;
        xp[p]=wrap_g(x+dt*pu,Lx);
        yp[p]=wrap_g(y+dt*pv,Ly);
        zp[p]=wrap_g(z+dt*pw,Lz);

        r2[p]=lpt_advance_r2_clamped(r2[p],r_ccn[p],Sp,growth_G,dt);
    }
}

template<typename TF>
void Lpt_droplets<TF>::prepare_device()
{
    if (!enabled) return;
    xp_g=cuda_vector<TF>(xp); yp_g=cuda_vector<TF>(yp); zp_g=cuda_vector<TF>(zp);
    up_g=cuda_vector<TF>(up); vp_g=cuda_vector<TF>(vp); wp_g=cuda_vector<TF>(wp);
    r2_g=cuda_vector<TF>(r2); r_ccn_g=cuda_vector<TF>(r_ccn); chi_local_g=cuda_vector<TF>(chi_local);
    chi_g.allocate(chi.size());
}

template<typename TF>
void Lpt_droplets<TF>::clear_device()
{
    if (!enabled) return;
    // Keep restart/output policy explicit.  Pull the diagnostics/state back here.
    xp=xp_g.to_vector(); yp=yp_g.to_vector(); zp=zp_g.to_vector();
    up=up_g.to_vector(); vp=vp_g.to_vector(); wp=wp_g.to_vector();
    r2=r2_g.to_vector(); r_ccn=r_ccn_g.to_vector(); chi_local=chi_local_g.to_vector();
    xp_g.free(); yp_g.free(); zp_g.free(); up_g.free(); vp_g.free(); wp_g.free();
    r2_g.free(); r_ccn_g.free(); chi_local_g.free(); chi_g.free();
}

template<typename TF>
void Lpt_droplets<TF>::exec_gpu(double dtin, const TF* u, const TF* v, const TF* w, const TF* s)
{
    const auto& gd=grid.get_grid_data();
    const TF dz=gd.zsize/TF(gd.ktot);
    constexpr int bs=128;
    const int ng=(gd.itot*gd.jtot*gd.ktot+bs-1)/bs;
    calc_chi_kernel<<<ng,bs>>>(chi_g.data(),s,kappa_T,gd.dx,gd.dy,dz,
                              gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    const int npblocks=int((np+bs-1)/bs);
    lpt_kernel<<<npblocks,bs>>>(np,xp_g.data(),yp_g.data(),zp_g.data(),up_g.data(),vp_g.data(),wp_g.data(),
        r2_g.data(),r_ccn_g.data(),chi_local_g.data(),u,v,w,s,chi_g.data(),TF(dtin),growth_G,rho_l,mu_air,gravity,
        gd.xsize,gd.ysize,gd.zsize,gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
        gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
}

#ifdef FLOAT_SINGLE
template void Lpt_droplets<float>::prepare_device();
template void Lpt_droplets<float>::clear_device();
template void Lpt_droplets<float>::exec_gpu(double,const float*,const float*,const float*,const float*);
#else
template void Lpt_droplets<double>::prepare_device();
template void Lpt_droplets<double>::clear_device();
template void Lpt_droplets<double>::exec_gpu(double,const double*,const double*,const double*,const double*);
#endif

#endif
