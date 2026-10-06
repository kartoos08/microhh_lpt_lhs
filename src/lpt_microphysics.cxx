#include "lpt_microphysics.h"

// Explicit instantiation keeps the CPU clamp available to unit tests and CPU builds.
#ifdef FLOAT_SINGLE
template Lpt::Growth_result<float> Lpt::grow_r2_clamped(
        float, float, float, float, const Lpt::Growth_config<float>&);
template float Lpt::stokes_response_time(float, const Lpt::Growth_config<float>&);
#else
template Lpt::Growth_result<double> Lpt::grow_r2_clamped(
        double, double, double, double, const Lpt::Growth_config<double>&);
template double Lpt::stokes_response_time(double, const Lpt::Growth_config<double>&);
#endif
