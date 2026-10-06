#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "entrainment_initializer.h"
#include "field3d.h"
#include "fields.h"
#include "grid.h"
#include "input.h"

namespace Lpt
{
namespace
{
inline std::uint64_t mix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template<typename TF>
TF tiebreak(const int i, const int j, const int k, const int seed)
{
    const std::uint64_t key =
        (std::uint64_t(std::uint32_t(i))      ) ^
        (std::uint64_t(std::uint32_t(j)) << 21) ^
        (std::uint64_t(std::uint32_t(k)) << 42) ^
        std::uint64_t(std::uint32_t(seed));
    const TF u = TF((mix64(key) >> 11) * (1.0/9007199254740992.0));
    return TF(1.e-10)*(u-TF(0.5));
}
}

template<typename TF>
void apply_entrainment_initial_condition(Grid<TF>& grid, Fields<TF>& fields, Input& input)
{
    const std::string scalar = input.get_item<std::string>("lpt", "scalar", "", "s");
    const TF s_cloud = input.get_item<TF>("entrainment", "S_cloud", "", TF(0));
    const TF s_sub = input.get_item<TF>("entrainment", "S_sub", "", TF(-0.2));
    const TF f_ent = input.get_item<TF>("entrainment", "f_ent", "", TF(0.3));
    const TF L0 = input.get_item<TF>("entrainment", "L0", "", TF(0.2));
    const int seed = input.get_item<int>("entrainment", "seed", "", 24680);

    if (!(f_ent >= TF(0) && f_ent <= TF(1)))
        throw std::runtime_error("[entrainment] f_ent must lie in [0,1]");
    if (!(L0 > TF(0)))
        throw std::runtime_error("[entrainment] L0 must be positive");

    auto& s = fields.sp.at(scalar)->fld;
    const auto& gd = grid.get_grid_data();
    const std::size_t ncell = std::size_t(gd.imax)*gd.jmax*gd.kmax;
    const std::size_t ndry = std::size_t(std::llround(double(f_ent)*double(ncell)));

    if (ndry == 0)
    {
        for (int k=gd.kstart; k<gd.kend; ++k)
            for (int j=gd.jstart; j<gd.jend; ++j)
                for (int i=gd.istart; i<gd.iend; ++i)
                    s[i + j*gd.icells + k*gd.ijcells] = s_cloud;
        return;
    }
    if (ndry == ncell)
    {
        for (int k=gd.kstart; k<gd.kend; ++k)
            for (int j=gd.jstart; j<gd.jend; ++j)
                for (int i=gd.istart; i<gd.iend; ++i)
                    s[i + j*gd.icells + k*gd.ijcells] = s_sub;
        return;
    }

    // A deterministic tri-periodic level-set with wavelength nearest to L0.
    // The quantile threshold enforces the requested dry volume fraction exactly
    // to one grid cell. The tiny hash perturbation breaks cosine-value ties.
    const int mx = std::max(1, int(std::lround(double(gd.xsize/L0))));
    const int my = std::max(1, int(std::lround(double(gd.ysize/L0))));
    const int mz = std::max(1, int(std::lround(double(gd.zsize/L0))));
    constexpr TF twopi = TF(6.2831853071795864769);
    const TF phx = TF(0.000123)*TF(seed % 7919);
    const TF phy = TF(0.000173)*TF(seed % 6841);
    const TF phz = TF(0.000211)*TF(seed % 5741);

    auto levelset = [&](int i, int j, int k)
    {
        const TF x = gd.x[i];
        const TF y = gd.y[j];
        const TF z = gd.z[k];
        return std::cos(twopi*TF(mx)*x/gd.xsize + phx)
             + std::cos(twopi*TF(my)*y/gd.ysize + phy)
             + std::cos(twopi*TF(mz)*z/gd.zsize + phz)
             + tiebreak<TF>(i-gd.istart, j-gd.jstart, k-gd.kstart, seed);
    };

    std::vector<TF> phi;
    phi.reserve(ncell);
    for (int k=gd.kstart; k<gd.kend; ++k)
        for (int j=gd.jstart; j<gd.jend; ++j)
            for (int i=gd.istart; i<gd.iend; ++i)
                phi.push_back(levelset(i,j,k));

    std::nth_element(phi.begin(), phi.begin()+std::ptrdiff_t(ndry-1), phi.end(), std::greater<TF>());
    const TF threshold = phi[ndry-1];

    for (int k=gd.kstart; k<gd.kend; ++k)
        for (int j=gd.jstart; j<gd.jend; ++j)
            for (int i=gd.istart; i<gd.iend; ++i)
            {
                const int ijk = i + j*gd.icells + k*gd.ijcells;
                s[ijk] = (levelset(i,j,k) >= threshold) ? s_sub : s_cloud;
            }
}

#ifdef FLOAT_SINGLE
template void apply_entrainment_initial_condition<float>(Grid<float>&, Fields<float>&, Input&);
#else
template void apply_entrainment_initial_condition<double>(Grid<double>&, Fields<double>&, Input&);
#endif
}
