# Triply-periodic prerequisite

The reference extension deliberately does **not** pretend that stock MicroHH's vertical atmospheric boundary/pressure treatment can be switched to periodic with an INI keyword.

Before using `base_case.ini` as a true tri-periodic DNS, your MicroHH branch must provide:

1. Uniform z spacing and wrap-around z indexing for all prognostic fields.
2. Periodic z ghost fills for scalar and all staggered momentum locations.
3. A pressure projection solving the periodic 3-D Poisson equation (zero mode explicitly fixed).
4. No wall/surface/Monin-Obukhov boundary tendencies.
5. A divergence regression test after the pressure projection.

The supplied LPT and spectral-forcing kernels already index z periodically and therefore assume items 1-4. If your branch already has a homogeneous-box/periodic-pressure implementation, connect `[homogeneous_box] periodic_z=true` to that implementation. Otherwise, this pressure/boundary change is a prerequisite beyond the two requested LPT/microphysics modifications.
