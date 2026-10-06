#include "lpt_droplets.h"
#include "lpt_microphysics.h"
#include "master.h"
#include "input.h"
#include "grid.h"
#include "fields.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
    template<typename TF>
    inline TF wrap_pos(TF x, TF L)
    {
        x -= L*std::floor(x/L);
        return (x >= L) ? TF(0) : x;
    }

    inline std::uint64_t splitmix64(std::uint64_t x)
    {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    template<typename TF>
    inline TF u01(std::uint64_t x)
    {
        constexpr double inv = 1.0/9007199254740992.0; // 2^-53
        return TF(((splitmix64(x) >> 11) + 0.5)*inv);
    }

    template<typename TF>
    inline TF normal01(std::uint64_t a, std::uint64_t b)
    {
        const TF u1 = std::max(u01<TF>(a), std::numeric_limits<TF>::min());
        const TF u2 = u01<TF>(b);
        return std::sqrt(TF(-2)*std::log(u1))*std::cos(TF(6.283185307179586476925286766559)*u2);
    }

    inline int imod(int i, int n)
    {
        i %= n;
        return i < 0 ? i+n : i;
    }

    template<typename TF>
    inline void lagrange6_weights(const TF x, TF* w)
    {
        // Degree-five Lagrange polynomial at nodes {-2,-1,0,1,2,3}; x in [0,1).
        w[0] = -x*(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(1))/TF(120);
        w[1] =  x*(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(2))/TF(24);
        w[2] = -(x-TF(3))*(x-TF(2))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(12);
        w[3] =  x*(x-TF(3))*(x-TF(2))*(x+TF(1))*(x+TF(2))/TF(12);
        w[4] = -x*(x-TF(3))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(24);
        w[5] =  x*(x-TF(2))*(x-TF(1))*(x+TF(1))*(x+TF(2))/TF(120);
    }

    template<typename TF>
    inline TF interp6_periodic(
            const TF* f, TF x, TF y, TF z,
            TF dx, TF dy, TF dz, int nx, int ny, int nz,
            int istart, int jstart, int kstart, int jj, int kk,
            // 0 = cell centre, 1 = face in the corresponding direction.
            int lx, int ly, int lz)
    {
        // MicroHH Grid_data loc uses 0 for centre and 1 for face.  For a
        // uniform box, centres are (i+1/2)dx and faces are i*dx.
        const TF sx = lx ? TF(0) : TF(0.5);
        const TF sy = ly ? TF(0) : TF(0.5);
        const TF sz = lz ? TF(0) : TF(0.5);
        const TF qx = x/dx-sx;
        const TF qy = y/dy-sy;
        const TF qz = z/dz-sz;
        const int i0 = int(std::floor(qx));
        const int j0 = int(std::floor(qy));
        const int k0 = int(std::floor(qz));
        TF wx[6], wy[6], wz[6];
        lagrange6_weights(qx-TF(i0), wx);
        lagrange6_weights(qy-TF(j0), wy);
        lagrange6_weights(qz-TF(k0), wz);
        TF ans = TF(0);
        for (int c=0; c<6; ++c)
        for (int b=0; b<6; ++b)
        for (int a=0; a<6; ++a)
        {
            const int i = istart + imod(i0+a-2, nx);
            const int j = jstart + imod(j0+b-2, ny);
            const int k = kstart + imod(k0+c-2, nz);
            ans += wx[a]*wy[b]*wz[c]*f[i + j*jj + k*kk];
        }
        return ans;
    }
}

template<typename TF>
Lpt_droplets<TF>::Lpt_droplets(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& input) :
    master(masterin), grid(gridin), fields(fieldsin)
{
    enabled = input.get_item<bool>("lpt", "enabled", "", false);
    particle_mode = input.get_item<std::string>("lpt", "particle_mode", "", "fixed_weighted");
    const int np_fixed = input.get_item<int>("lpt", "np_fixed", "", 12500000);
    const int strict_np = input.get_item<int>("lpt", "strict_np", "", np_fixed);
    np = std::size_t(particle_mode == "strict_one_to_one" ? strict_np : np_fixed);
    particle_weight = input.get_item<TF>("lpt", "particle_weight", "", TF(1));
    seed = std::uint64_t(input.get_item<int>("lpt", "seed", "", 271828));
    growth_G = input.get_item<TF>("lpt", "G", "", TF(1.e-10));
    rho_l = input.get_item<TF>("lpt", "rho_l", "", TF(1000));
    mu_air = input.get_item<TF>("lpt", "mu_air", "", TF(1.81e-5));
    kappa_T = input.get_item<TF>("lpt", "kappa_T", "", TF(2.e-5));
    gravity = input.get_item<bool>("lpt", "gravity", "", false);
    r_ccn_median = input.get_item<TF>("lpt", "r_ccn_median", "", TF(50.e-9));
    r_ccn_sigma_g = input.get_item<TF>("lpt", "r_ccn_sigma_g", "", TF(1.4));
    r_init_mode = input.get_item<std::string>("lpt", "r_init_mode", "", "fixed");
    r_init_fixed = input.get_item<TF>("lpt", "r_init_fixed", "", TF(10.e-6));
    kohler_activation_multiplier = input.get_item<TF>("lpt", "kohler_activation_multiplier", "", TF(20));
    kappa_ccn = input.get_item<TF>("lpt", "kappa_ccn", "", TF(0.3));

    const int order = input.get_item<int>("lpt", "interpolation_order", "", 6);
    if (enabled && order != 6)
        throw std::runtime_error("LPT extension currently requires interpolation_order=6");
    if (enabled && np == 0)
        throw std::runtime_error("LPT particle count must be positive");
}

template<typename TF>
void Lpt_droplets<TF>::init()
{
    if (!enabled) return;
    const auto& gd = grid.get_grid_data();
    if (gd.imax != gd.itot || gd.jmax != gd.jtot)
        throw std::runtime_error("This LPT reference extension is single-rank only; use npx=npy=1.");
    xp.resize(np); yp.resize(np); zp.resize(np);
    up.resize(np); vp.resize(np); wp.resize(np);
    r2.resize(np); r_ccn.resize(np); chi_local.resize(np);
    chi.resize(gd.ncells, TF(0));
    master.print_message("LPT: allocated %zu droplets, particle weight=%g\n", np, double(particle_weight));
}

template<typename TF>
TF Lpt_droplets<TF>::initial_wet_radius(const TF rd) const
{
    if (r_init_mode != "kohler_critical_scaled")
        return std::max(rd, r_init_fixed);

    // kappa-Koehler critical wet radius r* = sqrt(3*kappa*rd^3/A).
    // A is evaluated at 283 K with sigma_w=0.072 N/m.  This initialization
    // coupling makes kappa an active LHS dimension; the subsequent requested
    // evaporation law remains dr^2/dt=2GS with a dry-radius floor.
    const TF sigma_w = TF(0.072);
    const TF Mw = TF(0.01801528);
    const TF R = TF(8.314462618);
    const TF T = TF(283.0);
    const TF A = TF(2)*sigma_w*Mw/(R*T*rho_l);
    const TF rcrit = std::sqrt(std::max(TF(0), TF(3)*kappa_ccn*rd*rd*rd/A));
    return std::max(rd, kohler_activation_multiplier*rcrit);
}

template<typename TF>
void Lpt_droplets<TF>::initialize_host_particles()
{
    const auto& gd = grid.get_grid_data();
    const TF log_sigma = std::log(r_ccn_sigma_g);
    const TF* s0 = nullptr;
    auto its = fields.sp.find("s");
    if (its != fields.sp.end()) s0 = its->second->fld.data();
    #pragma omp parallel for schedule(static)
    for (std::int64_t pp=0; pp<std::int64_t(np); ++pp)
    {
        const std::size_t p=std::size_t(pp);
        // Deterministic rejection into the initially cloudy (S >= 0) portion.
        // With f_ent <= 0.5, the expected candidate count is <= 2.
        std::uint64_t key = seed + 64ULL*p;
        bool accepted = false;
        for (int trial=0; trial<64 && !accepted; ++trial)
        {
            const std::uint64_t q = key + 5ULL*trial;
            xp[p] = gd.xsize*u01<TF>(q+0);
            yp[p] = gd.ysize*u01<TF>(q+1);
            zp[p] = gd.zsize*u01<TF>(q+2);
            if (!s0) accepted = true;
            else
            {
                const int ii = std::min(gd.itot-1, int(xp[p]/gd.dx));
                const int jj0 = std::min(gd.jtot-1, int(yp[p]/gd.dy));
                const TF dz = gd.zsize/TF(gd.ktot);
                const int kk0 = std::min(gd.ktot-1, int(zp[p]/dz));
                const int idx=(gd.istart+ii)+(gd.jstart+jj0)*gd.jstride+(gd.kstart+kk0)*gd.kstride;
                accepted = s0[idx] >= TF(-1.e-14);
            }
        }
        if (!accepted)
        {
            // Probability <= (0.5)^64 for the requested f_ent range. Use a
            // deterministic centre fallback rather than throwing inside OpenMP.
            xp[p]=TF(0.5)*gd.xsize; yp[p]=TF(0.5)*gd.ysize; zp[p]=TF(0.5)*gd.zsize;
        }
        up[p] = vp[p] = wp[p] = TF(0);
        const TF zeta = normal01<TF>(key+51, key+52);
        r_ccn[p] = r_ccn_median*std::exp(log_sigma*zeta);
        const TF rw = initial_wet_radius(r_ccn[p]);
        r2[p] = rw*rw;
        chi_local[p] = TF(0);
    }
}

template<typename TF>
void Lpt_droplets<TF>::create()
{
    if (!enabled) return;
    initialize_host_particles();
}

template<typename TF>
void Lpt_droplets<TF>::calc_chi_cpu(const TF* s)
{
    const auto& gd = grid.get_grid_data();
    const TF dz = gd.zsize/TF(gd.ktot); // homogeneous-box extension requires uniform z
    for (int kk0=0; kk0<gd.ktot; ++kk0)
    for (int jj0=0; jj0<gd.jtot; ++jj0)
    for (int ii0=0; ii0<gd.itot; ++ii0)
    {
        auto at = [&](int di, int dj, int dk) -> TF {
            const int i = gd.istart + imod(ii0+di, gd.itot);
            const int j = gd.jstart + imod(jj0+dj, gd.jtot);
            const int k = gd.kstart + imod(kk0+dk, gd.ktot);
            return s[i + j*gd.jstride + k*gd.kstride];
        };
        // Fourth-order centred derivative. grad(s') == grad(s) for a homogeneous box.
        const TF sx = (-at(2,0,0)+TF(8)*at(1,0,0)-TF(8)*at(-1,0,0)+at(-2,0,0))/(TF(12)*gd.dx);
        const TF sy = (-at(0,2,0)+TF(8)*at(0,1,0)-TF(8)*at(0,-1,0)+at(0,-2,0))/(TF(12)*gd.dy);
        const TF sz = (-at(0,0,2)+TF(8)*at(0,0,1)-TF(8)*at(0,0,-1)+at(0,0,-2))/(TF(12)*dz);
        const int i = gd.istart+ii0, j = gd.jstart+jj0, k = gd.kstart+kk0;
        chi[i+j*gd.jstride+k*gd.kstride] = TF(2)*kappa_T*(sx*sx+sy*sy+sz*sz);
    }
}

template<typename TF>
void Lpt_droplets<TF>::exec_cpu(double dtin, const TF* u, const TF* v, const TF* w, const TF* s)
{
    const auto& gd = grid.get_grid_data();
    const TF dt = TF(dtin);
    const TF dz = gd.zsize/TF(gd.ktot);
    #pragma omp parallel for schedule(static)
    for (std::int64_t pp=0; pp<std::int64_t(np); ++pp)
    {
        const std::size_t p = std::size_t(pp);
        const TF uf = interp6_periodic(u,xp[p],yp[p],zp[p],gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
                                      gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride,1,0,0);
        const TF vf = interp6_periodic(v,xp[p],yp[p],zp[p],gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
                                      gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride,0,1,0);
        const TF wf = interp6_periodic(w,xp[p],yp[p],zp[p],gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
                                      gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride,0,0,1);
        const TF Sp = interp6_periodic(s,xp[p],yp[p],zp[p],gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
                                      gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride,0,0,0);
        chi_local[p] = interp6_periodic(chi.data(),xp[p],yp[p],zp[p],gd.dx,gd.dy,dz,gd.itot,gd.jtot,gd.ktot,
                                        gd.istart,gd.jstart,gd.kstart,gd.jstride,gd.kstride,0,0,0);

        // Stokes drag, solved analytically over the split step.
        const TF taup = std::max(TF(2)*rho_l*r2[p]/(TF(9)*mu_air), TF(1.e-12));
        const TF a = std::exp(-dt/taup);
        up[p] = uf + (up[p]-uf)*a;
        vp[p] = vf + (vp[p]-vf)*a;
        wp[p] = wf + (wp[p]-wf)*a;
        if (gravity)
            wp[p] -= TF(9.80665)*taup*(TF(1)-a);

        xp[p] = wrap_pos(xp[p] + dt*up[p], gd.xsize);
        yp[p] = wrap_pos(yp[p] + dt*vp[p], gd.ysize);
        zp[p] = wrap_pos(zp[p] + dt*wp[p], gd.zsize);

        // Requested physical clamp: no droplet can evaporate through its own dry core.
        r2[p] = lpt_advance_r2_clamped(r2[p], r_ccn[p], Sp, growth_G, dt);
    }
}

template<typename TF>
void Lpt_droplets<TF>::exec(double dt)
{
    if (!enabled) return;
    auto isu = fields.mp.find("u"), isv = fields.mp.find("v"), isw = fields.mp.find("w");
    auto iss = fields.sp.find("s");
    if (isu==fields.mp.end() || isv==fields.mp.end() || isw==fields.mp.end() || iss==fields.sp.end())
        throw std::runtime_error("LPT requires momentum fields u,v,w and prognostic scalar s");

    #ifdef USECUDA
    exec_gpu(dt, isu->second->fld_g.data(), isv->second->fld_g.data(), isw->second->fld_g.data(), iss->second->fld_g.data());
    #else
    calc_chi_cpu(iss->second->fld.data());
    exec_cpu(dt, isu->second->fld.data(), isv->second->fld.data(), isw->second->fld.data(), iss->second->fld.data());
    #endif
}

template<typename TF>
unsigned long Lpt_droplets<TF>::get_time_limit(unsigned long idt, double) const
{
    // Drag uses the exact exponential and growth is clamped, so no additional
    // stability restriction is required.  Returning idt leaves Eulerian CFL/DN
    // controls in charge.  Add an accuracy limit here if required.
    return idt;
}

#ifdef FLOAT_SINGLE
template class Lpt_droplets<float>;
#else
template class Lpt_droplets<double>;
#endif
