/*
Integration adapter for MicroHH 2.0.x.

This file intentionally isolates project-specific wiring from the numerics in
lpt_droplet.{h,cu}. Upstream MicroHH 2.0.2 does not expose a documented native
Lagrangian droplet API, so adapt these hooks to your fork's Master/Fields/Grid
interfaces rather than editing thermodynamics blindly.

Recommended call sequence per RK substep:
  1. obtain cell-centered u,v,w and supersaturation s arrays;
  2. compute chi_grid = 2*kappa_T*|grad(s')|^2;
  3. interpolate u,v,w,S,chi_grid to particles with the 6-point stencil;
  4. advance particle kinematics / drag;
  5. update r^2 <- max(r_ccn^2, r^2 + 2 G S_p dt_stage);
  6. accumulate diagnostics / two-way source terms if enabled.

Do not use a hard-coded global epsilon to define chi_local; chi_local is a
particle-sampled scalar-dissipation diagnostic.
*/

#include "lpt_droplet.h"
#include <stdexcept>

namespace microhh_lpt {

template<typename TF>
struct LptRuntimeConfig
{
    std::size_t np = 12500000;
    TF G = TF(1.0e-10);       // m^2 s^-1; replace with thermodynamic G(T,p)
    TF kappa_T = TF(2.0e-5);  // scalar diffusivity used in chi
    TF kappa_ccn = TF(0.3);
    bool kappa_activation = true;
    bool fixed_np_weighted = true;
};

// IMPORTANT ADAPTER POINT:
// In your MicroHH fork, make this class own device/host arrays and expose
// exec_substep(...) from the model time integrator. The flat-array interface
// below deliberately avoids depending on unstable internal class names.
template<typename TF>
class LptDropletExtension
{
public:
    explicit LptDropletExtension(LptRuntimeConfig<TF> cfg) : cfg_(cfg) {}

    void initialize(std::size_t n)
    {
        p_.resize(n);
        for (std::size_t q=0; q<n; ++q) p_.id[q]=q;
    }

    ParticleSoA<TF>& particles() noexcept { return p_; }
    const ParticleSoA<TF>& particles() const noexcept { return p_; }

    void update_microphysics_cpu(const std::vector<TF>& S_local, TF dt)
    {
        if (S_local.size()!=p_.size()) throw std::runtime_error("S_local size mismatch");
        evaporate_clamped_cpu(p_.r2.data(), p_.r_ccn.data(), S_local.data(), p_.size(), cfg_.G, dt);
    }

private:
    LptRuntimeConfig<TF> cfg_;
    ParticleSoA<TF> p_;
};

template class LptDropletExtension<float>;
template class LptDropletExtension<double>;

} // namespace microhh_lpt
