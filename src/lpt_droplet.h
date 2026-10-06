#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

#ifdef USECUDA
#include <cuda_runtime.h>
#endif

namespace microhh_lpt {

template<typename TF>
struct ParticleSoA
{
    std::vector<TF> x, y, z;
    std::vector<TF> up, vp, wp;
    std::vector<TF> r2;
    std::vector<TF> r_ccn;
    std::vector<TF> chi_local;
    std::vector<TF> weight;
    std::vector<std::uint64_t> id;

    void resize(std::size_t n)
    {
        x.resize(n); y.resize(n); z.resize(n);
        up.resize(n); vp.resize(n); wp.resize(n);
        r2.resize(n); r_ccn.resize(n); chi_local.resize(n);
        weight.resize(n, TF(1)); id.resize(n);
    }

    std::size_t size() const noexcept { return x.size(); }
};

template<typename TF>
inline void evaporate_clamped_cpu(
    TF* __restrict__ r2,
    const TF* __restrict__ r_ccn,
    const TF* __restrict__ S_local,
    std::size_t np,
    TF G,
    TF dt)
{
    for (std::size_t p = 0; p < np; ++p)
    {
        const TF floor_r2 = r_ccn[p] * r_ccn[p];
        const TF trial = r2[p] + TF(2) * G * S_local[p] * dt;
        r2[p] = std::max(floor_r2, trial);
    }
}

// Exact 6-point Lagrange interpolation on a uniform periodic mesh.
// This is 5th-degree / 6-point interpolation. If your project uses the phrase
// "6th-order" to mean six-point interpolation, this is the desired stencil.
// For formal O(dx^6) midpoint accuracy you can replace weights centrally.
template<typename TF>
inline void lagrange6_weights(TF xi, TF w[6])
{
    // nodes m = -2,-1,0,1,2,3, xi measured relative to node 0
    constexpr int node[6] = {-2,-1,0,1,2,3};
    for (int a = 0; a < 6; ++a)
    {
        TF wa = TF(1);
        for (int b = 0; b < 6; ++b)
            if (a != b)
                wa *= (xi - TF(node[b])) / TF(node[a] - node[b]);
        w[a] = wa;
    }
}

inline int pwrap(int i, int n)
{
    i %= n;
    return (i < 0) ? i + n : i;
}

template<typename TF>
inline TF interp6_periodic_cpu(
    const TF* __restrict__ f,
    int nx, int ny, int nz,
    TF x, TF y, TF z,
    TF dx, TF dy, TF dz)
{
    const TF gx = x/dx, gy = y/dy, gz = z/dz;
    const int i0 = static_cast<int>(std::floor(gx));
    const int j0 = static_cast<int>(std::floor(gy));
    const int k0 = static_cast<int>(std::floor(gz));
    TF wx[6], wy[6], wz[6];
    lagrange6_weights(gx-TF(i0), wx);
    lagrange6_weights(gy-TF(j0), wy);
    lagrange6_weights(gz-TF(k0), wz);

    TF out = TF(0);
    for (int c=0; c<6; ++c)
        for (int b=0; b<6; ++b)
            for (int a=0; a<6; ++a)
            {
                const int i = pwrap(i0 + a - 2, nx);
                const int j = pwrap(j0 + b - 2, ny);
                const int k = pwrap(k0 + c - 2, nz);
                out += wx[a]*wy[b]*wz[c]*f[(k*ny + j)*nx + i];
            }
    return out;
}

template<typename TF>
inline void compute_chi_grid_periodic_cpu(
    const TF* __restrict__ s,
    TF* __restrict__ chi,
    int nx, int ny, int nz,
    TF dx, TF dy, TF dz,
    TF kappa_T)
{
    // 6th-order centered derivative: (f[-3]-9f[-2]+45f[-1]-45f[1]+9f[2]-f[3]) / (60 dx)
    // sign is immaterial because the gradient is squared.
    for (int k=0; k<nz; ++k)
        for (int j=0; j<ny; ++j)
            for (int i=0; i<nx; ++i)
            {
                auto F = [&](int ii,int jj,int kk)->TF {
                    return s[(pwrap(kk,nz)*ny + pwrap(jj,ny))*nx + pwrap(ii,nx)];
                };
                const TF dsdx = (F(i-3,j,k)-TF(9)*F(i-2,j,k)+TF(45)*F(i-1,j,k)
                                -TF(45)*F(i+1,j,k)+TF(9)*F(i+2,j,k)-F(i+3,j,k))/(TF(60)*dx);
                const TF dsdy = (F(i,j-3,k)-TF(9)*F(i,j-2,k)+TF(45)*F(i,j-1,k)
                                -TF(45)*F(i,j+1,k)+TF(9)*F(i,j+2,k)-F(i,j+3,k))/(TF(60)*dy);
                const TF dsdz = (F(i,j,k-3)-TF(9)*F(i,j,k-2)+TF(45)*F(i,j,k-1)
                                -TF(45)*F(i,j,k+1)+TF(9)*F(i,j,k+2)-F(i,j,k+3))/(TF(60)*dz);
                chi[(k*ny+j)*nx+i] = TF(2)*kappa_T*(dsdx*dsdx + dsdy*dsdy + dsdz*dsdz);
            }
}

template<typename TF>
inline TF kohler_critical_radius(TF r_dry, TF kappa, TF T, TF sigma=TF(0.072),
                                 TF rho_w=TF(997.0), TF Rv=TF(461.5))
{
    // kappa-Koehler critical wet radius: r_c = sqrt(3*kappa*r_d^3/A),
    // A = 2 sigma/(rho_w Rv T). Used only for initialization/activation.
    const TF A = TF(2)*sigma/(rho_w*Rv*T);
    return std::sqrt(TF(3)*kappa*r_dry*r_dry*r_dry/A);
}

} // namespace microhh_lpt
