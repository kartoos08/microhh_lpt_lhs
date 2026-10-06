#ifndef LPT_DROPLETS_H
#define LPT_DROPLETS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "cuda_buffer.h"

class Master;
class Input;
template<typename> class Grid;
template<typename> class Fields;
template<typename> class Timeloop;

/*
 * Single-rank Eulerian-Lagrangian droplet extension for homogeneous-box DNS.
 *
 * State is structure-of-arrays for coalesced device access.  r2 is the wet
 * radius squared; r_ccn is the independent dry-core radius requested by the
 * experiment; chi_local is the particle-sampled scalar dissipation rate.
 *
 * IMPORTANT: this class assumes all three directions are periodic.  Stock
 * MicroHH is cyclic in x/y but not a triply-periodic homogeneous-box solver;
 * use this only with the companion z-periodic pressure/boundary extension.
 */
template<typename TF>
class Lpt_droplets
{
    public:
        Lpt_droplets(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Lpt_droplets() = default;

        void init();
        void create();
        void exec(double dt);
        unsigned long get_time_limit(unsigned long idt, double dt) const;

        #ifdef USECUDA
        void prepare_device();
        void clear_device();
        #endif

    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;

        bool enabled;
        bool gravity;
        std::string particle_mode;
        std::string r_init_mode;
        std::size_t np;
        std::uint64_t seed;

        TF particle_weight;
        TF growth_G;
        TF rho_l;
        TF mu_air;
        TF kappa_T;
        TF kappa_ccn;
        TF r_ccn_median;
        TF r_ccn_sigma_g;
        TF r_init_fixed;
        TF kohler_activation_multiplier;

        // Host state. vp* is particle velocity, not interpolated fluid velocity.
        std::vector<TF> xp, yp, zp;
        std::vector<TF> up, vp, wp;
        std::vector<TF> r2;
        std::vector<TF> r_ccn;
        std::vector<TF> chi_local;

        // Temporary cell-centred Eulerian chi field.
        std::vector<TF> chi;

        void initialize_host_particles();
        TF initial_wet_radius(TF rd) const;
        void calc_chi_cpu(const TF* s);
        void exec_cpu(double dt, const TF* u, const TF* v, const TF* w, const TF* s);

        #ifdef USECUDA
        cuda_vector<TF> xp_g, yp_g, zp_g;
        cuda_vector<TF> up_g, vp_g, wp_g;
        cuda_vector<TF> r2_g, r_ccn_g, chi_local_g;
        cuda_vector<TF> chi_g;
        void exec_gpu(double dt, const TF* u, const TF* v, const TF* w, const TF* s);
        #endif
};

#endif
