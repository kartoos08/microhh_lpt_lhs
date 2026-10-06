#ifdef USECUDA
#include "spectral_linear_force.h"
#include "grid.h"
#include "fields.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace
{
    template<typename TF> struct Ft;
    template<> struct Ft<float>
    {
        using C=cufftComplex;
        static constexpr cufftType r2c=CUFFT_R2C, c2r=CUFFT_C2R;
        static cufftResult fwd(cufftHandle p,float* a,C* b){return cufftExecR2C(p,a,b);}
        static cufftResult inv(cufftHandle p,C* a,float* b){return cufftExecC2R(p,a,b);}
    };
    template<> struct Ft<double>
    {
        using C=cufftDoubleComplex;
        static constexpr cufftType r2c=CUFFT_D2Z, c2r=CUFFT_Z2D;
        static cufftResult fwd(cufftHandle p,double* a,C* b){return cufftExecD2Z(p,a,b);}
        static cufftResult inv(cufftHandle p,C* a,double* b){return cufftExecZ2D(p,a,b);}
    };

    template<typename TF>
    __global__ void gather(const TF* src, TF* dst, int nx,int ny,int nz,int is,int js,int ks,int jj,int kk)
    {
        int q=blockIdx.x*blockDim.x+threadIdx.x; const int n=nx*ny*nz; if(q>=n)return;
        int i=q%nx, j=(q/nx)%ny, k=q/(nx*ny);
        dst[q]=src[(is+i)+(js+j)*jj+(ks+k)*kk];
    }

    template<typename TF>
    __global__ void add_tendency(const TF* src, TF* tend, TF invN, int nx,int ny,int nz,int is,int js,int ks,int jj,int kk)
    {
        int q=blockIdx.x*blockDim.x+threadIdx.x; const int n=nx*ny*nz; if(q>=n)return;
        int i=q%nx, j=(q/nx)%ny, k=q/(nx*ny);
        tend[(is+i)+(js+j)*jj+(ks+k)*kk] += src[q]*invN;
    }

    template<typename TF, typename C>
    __global__ void band_energy(const C* u,const C* v,const C* w,double* out,
                                 int nx,int ny,int nz,TF kmin,TF kmax,bool remove_mean)
    {
        const int nxh=nx/2+1; const int q=blockIdx.x*blockDim.x+threadIdx.x;
        const int nc=nxh*ny*nz; if(q>=nc)return;
        const int ix=q%nxh, iy=(q/nxh)%ny, iz=q/(nxh*ny);
        const int ky=(iy<=ny/2)?iy:iy-ny; const int kz=(iz<=nz/2)?iz:iz-nz;
        const TF km=sqrt(TF(ix*ix+ky*ky+kz*kz));
        if(km<kmin || km>kmax || (remove_mean && ix==0 && ky==0 && kz==0)) return;
        const double wt=(ix==0 || (nx%2==0 && ix==nx/2))?1.0:2.0;
        const double eu=double(u[q].x)*double(u[q].x)+double(u[q].y)*double(u[q].y);
        const double ev=double(v[q].x)*double(v[q].x)+double(v[q].y)*double(v[q].y);
        const double ew=double(w[q].x)*double(w[q].x)+double(w[q].y)*double(w[q].y);
        atomicAdd(out,wt*(eu+ev+ew));
    }

    template<typename TF, typename C>
    __global__ void make_forcing(C* a, TF alpha, int nx,int ny,int nz,TF kmin,TF kmax,bool remove_mean)
    {
        const int nxh=nx/2+1; const int q=blockIdx.x*blockDim.x+threadIdx.x;
        const int nc=nxh*ny*nz; if(q>=nc)return;
        const int ix=q%nxh, iy=(q/nxh)%ny, iz=q/(nxh*ny);
        const int ky=(iy<=ny/2)?iy:iy-ny; const int kz=(iz<=nz/2)?iz:iz-nz;
        const TF km=sqrt(TF(ix*ix+ky*ky+kz*kz));
        const bool keep=(km>=kmin && km<=kmax && !(remove_mean && ix==0 && ky==0 && kz==0));
        if(keep){a[q].x*=alpha; a[q].y*=alpha;} else {a[q].x=0; a[q].y=0;}
    }
}

template<typename TF>
struct Spectral_linear_force<TF>::Impl
{
    using C=typename Ft<TF>::C;
    cufftHandle pf=0,pi=0;
    cuda_vector<TF> ur,vr,wr;
    cuda_vector<C> uh,vh,wh;
    cuda_vector<double> energy;
};

template<typename TF>
void Spectral_linear_force<TF>::prepare_device()
{
    if(!enabled || impl) return;
    const auto& gd=grid.get_grid_data();
    if(gd.imax!=gd.itot || gd.jmax!=gd.jtot) throw std::runtime_error("spectral forcing reference backend is single-rank only");
    impl=new Impl;
    const std::size_t n=std::size_t(gd.itot)*gd.jtot*gd.ktot;
    const std::size_t nc=std::size_t(gd.itot/2+1)*gd.jtot*gd.ktot;
    impl->ur.allocate(n); impl->vr.allocate(n); impl->wr.allocate(n);
    impl->uh.allocate(nc); impl->vh.allocate(nc); impl->wh.allocate(nc); impl->energy.allocate(1);
    if(cufftPlan3d(&impl->pf,gd.ktot,gd.jtot,gd.itot,Ft<TF>::r2c)!=CUFFT_SUCCESS ||
       cufftPlan3d(&impl->pi,gd.ktot,gd.jtot,gd.itot,Ft<TF>::c2r)!=CUFFT_SUCCESS)
        throw std::runtime_error("cuFFT plan creation failed");
}

template<typename TF>
void Spectral_linear_force<TF>::clear_device()
{
    if(!impl)return;
    if(impl->pf)cufftDestroy(impl->pf); if(impl->pi)cufftDestroy(impl->pi);
    delete impl; impl=nullptr;
}

template<typename TF>
void Spectral_linear_force<TF>::exec()
{
    if(!enabled)return;
    if(!impl)throw std::runtime_error("spectral forcing device state not prepared");
    const auto& gd=grid.get_grid_data();
    auto u=fields.mp.at("u")->fld_g.data(); auto v=fields.mp.at("v")->fld_g.data(); auto w=fields.mp.at("w")->fld_g.data();
    auto ut=fields.mt.at("u")->fld_g.data(); auto vt=fields.mt.at("v")->fld_g.data(); auto wt=fields.mt.at("w")->fld_g.data();
    const int n=gd.itot*gd.jtot*gd.ktot, nc=(gd.itot/2+1)*gd.jtot*gd.ktot, bs=256;
    gather<<<(n+bs-1)/bs,bs>>>(u,impl->ur.data(),gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    gather<<<(n+bs-1)/bs,bs>>>(v,impl->vr.data(),gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    gather<<<(n+bs-1)/bs,bs>>>(w,impl->wr.data(),gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    if(Ft<TF>::fwd(impl->pf,impl->ur.data(),impl->uh.data())!=CUFFT_SUCCESS ||
       Ft<TF>::fwd(impl->pf,impl->vr.data(),impl->vh.data())!=CUFFT_SUCCESS ||
       Ft<TF>::fwd(impl->pf,impl->wr.data(),impl->wh.data())!=CUFFT_SUCCESS) throw std::runtime_error("cuFFT forward failed");
    cudaMemset(impl->energy.data(),0,sizeof(double));
    band_energy<<<(nc+bs-1)/bs,bs>>>(impl->uh.data(),impl->vh.data(),impl->wh.data(),impl->energy.data(),gd.itot,gd.jtot,gd.ktot,kmin,kmax,remove_mean);
    double raw=0; cudaMemcpy(&raw,impl->energy.data(),sizeof(double),cudaMemcpyDeviceToHost);
    const double N=double(n);
    const double Ef=0.5*raw/(N*N);
    if(!(Ef>0.0)) throw std::runtime_error("spectral forcing band contains zero kinetic energy");
    const TF alpha=TF(double(epsilon)/(2.0*Ef));
    make_forcing<<<(nc+bs-1)/bs,bs>>>(impl->uh.data(),alpha,gd.itot,gd.jtot,gd.ktot,kmin,kmax,remove_mean);
    make_forcing<<<(nc+bs-1)/bs,bs>>>(impl->vh.data(),alpha,gd.itot,gd.jtot,gd.ktot,kmin,kmax,remove_mean);
    make_forcing<<<(nc+bs-1)/bs,bs>>>(impl->wh.data(),alpha,gd.itot,gd.jtot,gd.ktot,kmin,kmax,remove_mean);
    Ft<TF>::inv(impl->pi,impl->uh.data(),impl->ur.data()); Ft<TF>::inv(impl->pi,impl->vh.data(),impl->vr.data()); Ft<TF>::inv(impl->pi,impl->wh.data(),impl->wr.data());
    const TF invN=TF(1)/TF(n);
    add_tendency<<<(n+bs-1)/bs,bs>>>(impl->ur.data(),ut,invN,gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    add_tendency<<<(n+bs-1)/bs,bs>>>(impl->vr.data(),vt,invN,gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
    add_tendency<<<(n+bs-1)/bs,bs>>>(impl->wr.data(),wt,invN,gd.itot,gd.jtot,gd.ktot,gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride);
}

#ifdef FLOAT_SINGLE
template void Spectral_linear_force<float>::prepare_device();
template void Spectral_linear_force<float>::clear_device();
template void Spectral_linear_force<float>::exec();
#else
template void Spectral_linear_force<double>::prepare_device();
template void Spectral_linear_force<double>::clear_device();
template void Spectral_linear_force<double>::exec();
#endif
#endif
