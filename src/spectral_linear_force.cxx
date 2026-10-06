#include "spectral_linear_force.h"
#include "input.h"
#include "master.h"
#include "grid.h"
#include "fields.h"
#include <stdexcept>

template<typename TF>
Spectral_linear_force<TF>::Spectral_linear_force(Master& m, Grid<TF>& g, Fields<TF>& f, Input& input):
    master(m), grid(g), fields(f)
{
    enabled=input.get_item<bool>("spectral_force","enabled","",false);
    epsilon=input.get_item<TF>("spectral_force","epsilon","",TF(0));
    kmin=input.get_item<TF>("spectral_force","kmin","",TF(1));
    kmax=input.get_item<TF>("spectral_force","kmax","",TF(2));
    remove_mean=input.get_item<bool>("spectral_force","remove_mean","",true);
    if (enabled && !(epsilon>TF(0))) throw std::runtime_error("spectral_force epsilon must be positive");
}

template<typename TF>
Spectral_linear_force<TF>::~Spectral_linear_force()
{
    #ifdef USECUDA
    clear_device();
    #endif
}

template<typename TF>
void Spectral_linear_force<TF>::init()
{
    if (!enabled) return;
    #ifndef USECUDA
    throw std::runtime_error("Reference spectral_linear_force is CUDA-only; add an FFTW backend for CPU runs.");
    #endif
}

#ifndef USECUDA
template<typename TF> void Spectral_linear_force<TF>::exec() {}
#endif

#ifdef FLOAT_SINGLE
template class Spectral_linear_force<float>;
#else
template class Spectral_linear_force<double>;
#endif
