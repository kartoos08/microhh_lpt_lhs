#ifndef LPT_INTERP6_H
#define LPT_INTERP6_H

#include <cmath>

#ifdef __CUDACC__
#define LPT_HD __host__ __device__ inline
#else
#define LPT_HD inline
#endif

namespace Lpt
{

template<typename TF>
struct Periodic_field_view
{
    const TF* data;
    int nx, ny, nz;
    int istart, jstart, kstart;
    int jstride, kstride;
    TF dx, dy, dz;
    TF shift_x, shift_y, shift_z; // 0.5 for cell center, 0.0 for face.
};

LPT_HD int imod(const int a, const int n)
{
    const int r = a % n;
    return (r < 0) ? r + n : r;
}

template<typename TF>
LPT_HD void lagrange6_weights(const TF x, TF w[6], TF dw[6])
{
    // Six-point Lagrange interpolation on offsets {-2,-1,0,1,2,3}.
    // The polynomial is degree 5 and the interpolation error is O(h^6).
    constexpr int o[6] = {-2, -1, 0, 1, 2, 3};

    for (int a = 0; a < 6; ++a)
    {
        TF wa = TF(1);
        for (int b = 0; b < 6; ++b)
            if (b != a)
                wa *= (x - TF(o[b])) / TF(o[a] - o[b]);
        w[a] = wa;

        TF dwa = TF(0);
        for (int m = 0; m < 6; ++m)
        {
            if (m == a)
                continue;
            TF term = TF(1) / TF(o[a] - o[m]);
            for (int b = 0; b < 6; ++b)
            {
                if (b == a || b == m)
                    continue;
                term *= (x - TF(o[b])) / TF(o[a] - o[b]);
            }
            dwa += term;
        }
        dw[a] = dwa;
    }
}

template<typename TF>
LPT_HD TF sample6(const Periodic_field_view<TF>& f, const TF x, const TF y, const TF z)
{
    const TF gx = x / f.dx - f.shift_x;
    const TF gy = y / f.dy - f.shift_y;
    const TF gz = z / f.dz - f.shift_z;

    const int i0 = static_cast<int>(floor(gx));
    const int j0 = static_cast<int>(floor(gy));
    const int k0 = static_cast<int>(floor(gz));

    const TF xi = gx - TF(i0);
    const TF yi = gy - TF(j0);
    const TF zi = gz - TF(k0);

    TF wx[6], wy[6], wz[6], dummy[6];
    lagrange6_weights(xi, wx, dummy);
    lagrange6_weights(yi, wy, dummy);
    lagrange6_weights(zi, wz, dummy);

    TF value = TF(0);
    for (int c = 0; c < 6; ++c)
    {
        const int k = f.kstart + imod(k0 + c - 2, f.nz);
        for (int b = 0; b < 6; ++b)
        {
            const int j = f.jstart + imod(j0 + b - 2, f.ny);
            TF line = TF(0);
            for (int a = 0; a < 6; ++a)
            {
                const int i = f.istart + imod(i0 + a - 2, f.nx);
                const int ijk = i + j*f.jstride + k*f.kstride;
                line += wx[a] * f.data[ijk];
            }
            value += wz[c] * wy[b] * line;
        }
    }
    return value;
}

template<typename TF>
LPT_HD void sample6_value_gradient(
        const Periodic_field_view<TF>& f,
        const TF x, const TF y, const TF z,
        TF& value, TF& dfdx, TF& dfdy, TF& dfdz)
{
    const TF gx = x / f.dx - f.shift_x;
    const TF gy = y / f.dy - f.shift_y;
    const TF gz = z / f.dz - f.shift_z;

    const int i0 = static_cast<int>(floor(gx));
    const int j0 = static_cast<int>(floor(gy));
    const int k0 = static_cast<int>(floor(gz));

    const TF xi = gx - TF(i0);
    const TF yi = gy - TF(j0);
    const TF zi = gz - TF(k0);

    TF wx[6], wy[6], wz[6];
    TF dwx[6], dwy[6], dwz[6];
    lagrange6_weights(xi, wx, dwx);
    lagrange6_weights(yi, wy, dwy);
    lagrange6_weights(zi, wz, dwz);

    value = dfdx = dfdy = dfdz = TF(0);
    for (int c = 0; c < 6; ++c)
    {
        const int k = f.kstart + imod(k0 + c - 2, f.nz);
        for (int b = 0; b < 6; ++b)
        {
            const int j = f.jstart + imod(j0 + b - 2, f.ny);
            for (int a = 0; a < 6; ++a)
            {
                const int i = f.istart + imod(i0 + a - 2, f.nx);
                const int ijk = i + j*f.jstride + k*f.kstride;
                const TF q = f.data[ijk];
                value += wx[a]  * wy[b]  * wz[c]  * q;
                dfdx  += dwx[a] * wy[b]  * wz[c]  * q / f.dx;
                dfdy  += wx[a]  * dwy[b] * wz[c]  * q / f.dy;
                dfdz  += wx[a]  * wy[b]  * dwz[c] * q / f.dz;
            }
        }
    }
}

} // namespace Lpt

#endif
