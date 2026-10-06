# MicroHH LPT/LHS extension scaffold

Target baseline: MicroHH 2.0.x (verify against your pinned commit before compiling).

## What this contains
- `src/lpt_droplet.h`: extended particle SoA with `r_ccn` and `chi_local`, CPU 6-point interpolation, scalar-dissipation grid calculation, dry-radius evaporation clamp, optional kappa-Koehler activation helper.
- `src/lpt_droplet.cu`: CUDA kernels for chi, 6-point interpolation, and clamped evaporation.
- `src/lpt_adapter.cxx`: minimal adapter layer showing where to call the extension from a MicroHH RK substep.
- `cases/base_case.ini`: 256^3 template with extension keys for spectral forcing, entrainment, and LPT.
- `scripts/generate_lhs.py`: 6-D 100-point Latin Hypercube generator plus case manifests and SLURM array script.

## Critical integration notes
1. Upstream MicroHH 2.0.2 is C++/CUDA DNS/LES and supports moist physics, but this scaffold does **not** assume an upstream native Lagrangian droplet API. Wire `LptDropletExtension` into your fork's integrator/field access layer.
2. Stock MicroHH is horizontally periodic; a genuinely tri-periodic z direction requires a pressure/boundary implementation consistent with the Poisson solver. Do not fake z periodicity only inside particle interpolation.
3. The CUDA 6-point interpolation assumes logically uniform, cell-centered, fully periodic arrays. Staggered MicroHH velocity components require component-specific coordinate offsets before interpolation.
4. If velocity/scalar arrays include halos, change indexing to use interior strides/offsets rather than the compact nx*ny*nz indexing shown here.
5. `chi = 2*kappa_T*|grad(s')|^2`; subtracting a constant/plane mean does not change the gradient. If your mean varies spatially, explicitly form s' before differentiation.
6. `kappa_ccn` does not affect `dr^2/dt=2GS` with only a dry-radius floor. The helper therefore uses kappa-Koehler theory at activation/initialization; turn `kappa_activation=false` if kappa is intentionally metadata only.
7. Fixed 12.5M particles cannot simultaneously represent true 1:1 physical droplets while N0 varies. `generate_lhs.py` supports weighted fixed-Np and strict one-to-one modes.

## Generate cases
```bash
python scripts/generate_lhs.py --base cases/base_case.ini --out lhs_cases -n 100
sbatch lhs_cases/submit_array.slurm
```

## Suggested MicroHH integration
- Construct LPT extension after grid/fields are initialized.
- Initialize particle dry radius and wet radius; for kappa-dependent activation use `kohler_critical_radius` only as an initialization relation.
- Every RK stage: compute `chi_grid`, interpolate `u,v,w,S,chi`, advance trajectory, then clamp the r^2 update.
- For two-way thermodynamic coupling, accumulate condensational mass source conservatively to the Eulerian vapor/temperature equations with the same particle weights.
- Add restart I/O for `r_ccn`, `chi_local`, `r2`, particle coordinates/velocities, IDs, and weights.

## Compile
If your MicroHH CMake does not automatically pick up added source files, add the new `.cxx`/`.cu` explicitly to the main target. CUDA+MPI support is release/configuration dependent; verify your target machine setup against the pinned MicroHH documentation.
