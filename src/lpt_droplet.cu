#include "lpt_droplet.h"

#ifdef USECUDA
namespace microhh_lpt {

template<typename TF>
__device__ inline int pwrap_d(int i, int n)
{
    i %= n;
    return (i < 0) ? i+n : i;
}

template<typename TF>
__device__ inline void lagrange6_weights_d(TF xi, TF w[6])
{
    const int node[6] = {-2,-1,0,1,2,3};
    #pragma unroll
    for (int a=0; a<6; ++a)
    {
        TF wa = TF(1);
        #pragma unroll
        for (int b=0; b<6; ++b)
            if (a != b) wa *= (xi-TF(node[b]))/TF(node[a]-node[b]);
        w[a] = wa;
    }
}

template<typename TF>
__device__ inline TF interp6_periodic_device(const TF* f, int nx,int ny,int nz,
                                             TF x,TF y,TF z,TF dx,TF dy,TF dz)
{
    TF gx=x/dx, gy=y/dy, gz=z/dz;
    int i0=(int)floor(gx), j0=(int)floor(gy), k0=(int)floor(gz);
    TF wx[6], wy[6], wz[6];
    lagrange6_weights_d(gx-TF(i0),wx);
    lagrange6_weights_d(gy-TF(j0),wy);
    lagrange6_weights_d(gz-TF(k0),wz);
    TF out=TF(0);
    #pragma unroll
    for(int c=0;c<6;++c) {
        int k=pwrap_d<TF>(k0+c-2,nz);
        #pragma unroll
        for(int b=0;b<6;++b) {
            int j=pwrap_d<TF>(j0+b-2,ny);
            #pragma unroll
            for(int a=0;a<6;++a) {
                int i=pwrap_d<TF>(i0+a-2,nx);
                out += wx[a]*wy[b]*wz[c]*f[(k*ny+j)*nx+i];
            }
        }
    }
    return out;
}

template<typename TF>
__global__ void chi_grid_kernel(const TF* s, TF* chi, int nx,int ny,int nz,
                                TF dx,TF dy,TF dz,TF kappa_T)
{
    const int q = blockIdx.x*blockDim.x + threadIdx.x;
    const int n = nx*ny*nz;
    if(q>=n) return;
    const int i=q%nx, j=(q/nx)%ny, k=q/(nx*ny);
    auto F = [&](int ii,int jj,int kk)->TF {
        return s[(pwrap_d<TF>(kk,nz)*ny+pwrap_d<TF>(jj,ny))*nx+pwrap_d<TF>(ii,nx)];
    };
    const TF dsdx=(F(i-3,j,k)-TF(9)*F(i-2,j,k)+TF(45)*F(i-1,j,k)-TF(45)*F(i+1,j,k)+TF(9)*F(i+2,j,k)-F(i+3,j,k))/(TF(60)*dx);
    const TF dsdy=(F(i,j-3,k)-TF(9)*F(i,j-2,k)+TF(45)*F(i,j-1,k)-TF(45)*F(i,j+1,k)+TF(9)*F(i,j+2,k)-F(i,j+3,k))/(TF(60)*dy);
    const TF dsdz=(F(i,j,k-3)-TF(9)*F(i,j,k-2)+TF(45)*F(i,j,k-1)-TF(45)*F(i,j,k+1)+TF(9)*F(i,j,k+2)-F(i,j,k+3))/(TF(60)*dz);
    chi[q]=TF(2)*kappa_T*(dsdx*dsdx+dsdy*dsdy+dsdz*dsdz);
}

template<typename TF>
__global__ void interpolate_particles_kernel(
    const TF* u,const TF* v,const TF* w,const TF* S,const TF* chi_grid,
    const TF* x,const TF* y,const TF* z,
    TF* ui,TF* vi,TF* wi,TF* Si,TF* chi_local,
    std::size_t np,int nx,int ny,int nz,TF dx,TF dy,TF dz)
{
    const std::size_t p=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=np) return;
    ui[p]=interp6_periodic_device(u,nx,ny,nz,x[p],y[p],z[p],dx,dy,dz);
    vi[p]=interp6_periodic_device(v,nx,ny,nz,x[p],y[p],z[p],dx,dy,dz);
    wi[p]=interp6_periodic_device(w,nx,ny,nz,x[p],y[p],z[p],dx,dy,dz);
    Si[p]=interp6_periodic_device(S,nx,ny,nz,x[p],y[p],z[p],dx,dy,dz);
    chi_local[p]=interp6_periodic_device(chi_grid,nx,ny,nz,x[p],y[p],z[p],dx,dy,dz);
}

template<typename TF>
__global__ void evaporate_clamped_kernel(TF* r2,const TF* r_ccn,const TF* S,
                                         std::size_t np,TF G,TF dt)
{
    const std::size_t p=(std::size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=np) return;
    const TF floor_r2=r_ccn[p]*r_ccn[p];
    const TF trial=r2[p]+TF(2)*G*S[p]*dt;
    r2[p]=(trial>floor_r2)?trial:floor_r2;
}

// Explicit instantiations; keep only the precision(s) used by your build.
template __global__ void chi_grid_kernel<float>(const float*,float*,int,int,int,float,float,float,float);
template __global__ void chi_grid_kernel<double>(const double*,double*,int,int,int,double,double,double,double);
template __global__ void interpolate_particles_kernel<float>(const float*,const float*,const float*,const float*,const float*,const float*,const float*,const float*,float*,float*,float*,float*,float*,std::size_t,int,int,int,float,float,float);
template __global__ void interpolate_particles_kernel<double>(const double*,const double*,const double*,const double*,const double*,const double*,const double*,const double*,double*,double*,double*,double*,double*,std::size_t,int,int,int,double,double,double);
template __global__ void evaporate_clamped_kernel<float>(float*,const float*,const float*,std::size_t,float,float);
template __global__ void evaporate_clamped_kernel<double>(double*,const double*,const double*,std::size_t,double,double);

} // namespace microhh_lpt
#endif
