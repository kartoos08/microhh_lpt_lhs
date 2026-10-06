#ifndef SPECTRAL_LINEAR_FORCE_H
#define SPECTRAL_LINEAR_FORCE_H

#include "cuda_buffer.h"
#include <cstddef>

class Master;
class Input;
template<typename> class Grid;
template<typename> class Fields;

/* Low-k linear forcing f_hat = alpha u_hat, with alpha chosen each call so
 * <u.f> = epsilon.  This implementation is intended for a single-GPU,
 * triply-periodic homogeneous box. */
template<typename TF>
class Spectral_linear_force
{
    public:
        Spectral_linear_force(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Spectral_linear_force();
        void init();
        void exec();
        #ifdef USECUDA
        void prepare_device();
        void clear_device();
        #endif
    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;
        bool enabled;
        bool remove_mean;
        TF epsilon;
        TF kmin;
        TF kmax;
        #ifdef USECUDA
        struct Impl;
        Impl* impl=nullptr;
        #endif
};
#endif
