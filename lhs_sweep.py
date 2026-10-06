#!/usr/bin/env python3
"""Generate a 100-case 6-variable Latin-hypercube sweep for MicroHH LPT DNS.

The first variable (epsilon) is log-uniform; the remaining five are uniform in
physical space. A reproducible scipy.stats.qmc.LatinHypercube design is used.

This script writes:
  cases/case_000/...case_099/<case>.ini
  cases/.../manifest.json
  case_table.csv
  submit_array.slurm

Optionally it also writes minimal <case>_input.nc files when --make-netcdf is set.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class Sample:
    epsilon: float
    S_sub: float
    N0_cm3: float
    f_ent: float
    L0_m: float
    kappa_hygro: float


PARAM_BOUNDS = {
    "epsilon": (1.0e-4, 1.0e-2),
    "S_sub": (-0.50, -0.05),
    "N0_cm3": (50.0, 500.0),
    "f_ent": (0.10, 0.50),
    "L0_m": (0.05, 0.50),
    "kappa_hygro": (0.01, 0.60),
}


def parser_with_case() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#", ";"))
    cfg.optionxform = str
    return cfg


def setv(cfg, section: str, key: str, value) -> None:
    if not cfg.has_section(section):
        cfg.add_section(section)
    if isinstance(value, bool):
        cfg.set(section, key, "true" if value else "false")
    elif isinstance(value, int):
        cfg.set(section, key, str(value))
    elif isinstance(value, float):
        cfg.set(section, key, f"{value:.12g}")
    else:
        cfg.set(section, key, str(value))


def lhs_samples(n: int, seed: int) -> List[Sample]:
    engine = qmc.LatinHypercube(d=6, seed=seed, optimization="random-cd")
    u = engine.random(n)

    # epsilon spans two decades; sample uniformly in log10(epsilon).
    e0, e1 = np.log10(PARAM_BOUNDS["epsilon"])
    epsilon = 10.0 ** (e0 + u[:, 0] * (e1 - e0))

    def linear(col: int, key: str) -> np.ndarray:
        lo, hi = PARAM_BOUNDS[key]
        return lo + u[:, col] * (hi - lo)

    arrs = [
        epsilon,
        linear(1, "S_sub"),
        linear(2, "N0_cm3"),
        linear(3, "f_ent"),
        linear(4, "L0_m"),
        linear(5, "kappa_hygro"),
    ]
    return [Sample(*(float(a[i]) for a in arrs)) for i in range(n)]


def derived(sample: Sample, cfg, fixed_np: int, population_mode: str) -> Dict[str, object]:
    nu = cfg.getfloat("lpt", "nu_air", fallback=1.5e-5)
    rho_w = cfg.getfloat("lpt", "rho_w", fallback=997.0)
    rho_air = cfg.getfloat("lpt", "rho_air", fallback=1.2)
    r0 = cfg.getfloat("lpt", "r0", fallback=10e-6)
    rccn = cfg.getfloat("lpt", "r_ccn", fallback=0.05e-6)
    G = cfg.getfloat("lpt", "G", fallback=1e-10)
    Lx = cfg.getfloat("grid", "xsize")
    Ly = cfg.getfloat("grid", "ysize")
    Lz = cfg.getfloat("grid", "zsize")
    nx = cfg.getint("grid", "itot")
    ny = cfg.getint("grid", "jtot")
    nz = cfg.getint("grid", "ktot")

    V = Lx * Ly * Lz
    V_cloud = (1.0 - sample.f_ent) * V
    N0_m3 = sample.N0_cm3 * 1.0e6
    nphysical = N0_m3 * V_cloud

    if population_mode == "strict_one_to_one":
        nparticles = int(round(nphysical))
        particle_weight = 1.0
    else:
        nparticles = fixed_np
        particle_weight = nphysical / fixed_np

    eta = (nu**3 / sample.epsilon) ** 0.25
    tau_eta = math.sqrt(nu / sample.epsilon)
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    dmax = max(dx, dy, dz)

    tau_p0 = 2.0 * rho_w * r0**2 / (9.0 * rho_air * nu)
    St0 = tau_p0 / tau_eta
    tau_mix = (sample.L0_m**2 / sample.epsilon) ** (1.0 / 3.0)
    tau_evap = max(r0**2 - rccn**2, 0.0) / (2.0 * G * abs(sample.S_sub))
    Da_nominal = tau_mix / tau_evap if tau_evap > 0.0 else math.inf

    # Periodic-box wavelengths are discrete. This is the closest axis-aligned shell.
    n_shell = max(1, int(round(Lx / sample.L0_m)))
    L0_eff = Lx / n_shell

    warnings: List[str] = []
    if dmax > eta:
        warnings.append(
            f"Under-resolved Kolmogorov scale: max(dx,dy,dz)/eta={dmax/eta:.3g}. "
            "Treat this as DNS-style/no-SGS, not resolved DNS, unless the box/grid is changed."
        )
    if population_mode == "strict_one_to_one" and nparticles > 100_000_000:
        warnings.append(
            f"Strict one-to-one population is {nparticles:,} particles; this is far above the nominal 12.5M target."
        )
    if population_mode == "fixed_weighted" and abs(particle_weight - 1.0) > 1e-12:
        warnings.append(
            f"Fixed numerical population uses super-droplet weight={particle_weight:.6g}; it is not literal 1:1 physical sampling."
        )
    warnings.append(
        "kappa_hygro is sampled but inactive in the requested clamp-only dr^2/dt=2GS law. "
        "A kappa-Kohler equilibrium term is required for it to affect trajectories/survival."
    )
    if abs(L0_eff - sample.L0_m) / sample.L0_m > 0.05:
        warnings.append(
            f"Nearest simple periodic Fourier wavelength is {L0_eff:.6g} m, "
            f"{100*abs(L0_eff-sample.L0_m)/sample.L0_m:.1f}% from sampled L0."
        )

    return {
        "N0_m3": N0_m3,
        "V_cloud_m3": V_cloud,
        "nphysical_cloud_droplets": nphysical,
        "nparticles": nparticles,
        "particle_weight": particle_weight,
        "eta_m": eta,
        "tau_eta_s": tau_eta,
        "max_grid_spacing_m": dmax,
        "dx_over_eta": dmax / eta,
        "tau_p0_s": tau_p0,
        "St0": St0,
        "tau_mix_s": tau_mix,
        "tau_evap_to_ccn_s": tau_evap,
        "Da_nominal": Da_nominal,
        "St_times_Da_nominal": St0 * Da_nominal,
        "nearest_periodic_shell": n_shell,
        "L0_effective_simple_mode_m": L0_eff,
        "warnings": warnings,
    }


def write_ini(cfg, path: Path) -> None:
    with path.open("w") as fh:
        cfg.write(fh, space_around_delimiters=False)


def write_slurm(outroot: Path, n: int, args) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name=mh_lpt_lhs
#SBATCH --array=0-{n-1}%{args.max_concurrent}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gres=gpu:{args.gpus_per_task}
#SBATCH --time={args.walltime}
#SBATCH --output=slurm/%A_%a.out
#SBATCH --error=slurm/%A_%a.err

set -euo pipefail

ROOT=\"$(cd \"$(dirname \"$0\")\" && pwd)\"
CASE=$(printf \"case_%03d\" \"${{SLURM_ARRAY_TASK_ID}}\")
CDIR=\"${{ROOT}}/cases/${{CASE}}\"
mkdir -p \"${{ROOT}}/slurm\"
cd \"${{CDIR}}\"

MICROHH_BIN=\"${{MICROHH_BIN:-{args.microhh_bin}}}\"

# Current upstream MicroHH supports CUDA and MPI separately, not together;
# this array assumes one GPU / one MicroHH process per case.
\"${{MICROHH_BIN}}\" init \"${{CASE}}\"
\"${{MICROHH_BIN}}\" run  \"${{CASE}}\"
"""
    p = outroot / "submit_array.slurm"
    p.write_text(text)
    p.chmod(0o755)


def maybe_make_netcdf(case_dir: Path, case: str, cfg) -> None:
    try:
        import netCDF4 as nc4
    except ImportError as exc:
        raise RuntimeError("--make-netcdf requested but python-netCDF4 is unavailable") from exc

    nz = cfg.getint("grid", "ktot")
    lz = cfg.getfloat("grid", "zsize")
    dz = lz / nz
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz
    out = case_dir / f"{case}_input.nc"
    with nc4.Dataset(out, "w", format="NETCDF4") as ds:
        ds.createDimension("z", nz)
        zv = ds.createVariable("z", "f8", ("z",))
        zv.units = "m"
        zv[:] = z
        init = ds.createGroup("init")
        for name, units in (("u", "m s-1"), ("v", "m s-1"), ("s", "1")):
            var = init.createVariable(name, "f8", ("z",))
            var.units = units
            var[:] = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("base_case.ini"))
    ap.add_argument("--out", type=Path, default=Path("lhs_100"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--fixed-np", type=int, default=12_500_000)
    ap.add_argument(
        "--population-mode",
        choices=("fixed_weighted", "strict_one_to_one"),
        default="fixed_weighted",
        help="fixed_weighted preserves ~12.5M numerical particles; strict_one_to_one sets one numerical particle per physical droplet",
    )
    ap.add_argument("--make-netcdf", action="store_true")
    ap.add_argument("--microhh-bin", default="/path/to/microhh")
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--gpus-per-task", type=int, default=1)
    ap.add_argument("--walltime", default="24:00:00")
    args = ap.parse_args()

    base = parser_with_case()
    if not base.read(args.base):
        raise SystemExit(f"Could not read {args.base}")

    outroot = args.out.resolve()
    cases_root = outroot / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    (outroot / "slurm").mkdir(exist_ok=True)

    samples = lhs_samples(args.n, args.seed)
    rows = []

    for i, sample in enumerate(samples):
        case = f"case_{i:03d}"
        cdir = cases_root / case
        if cdir.exists():
            shutil.rmtree(cdir)
        cdir.mkdir(parents=True)

        cfg = parser_with_case()
        cfg.read(args.base)
        diag = derived(sample, cfg, args.fixed_np, args.population_mode)

        setv(cfg, "spectral_forcing", "epsilon_target", sample.epsilon)
        setv(cfg, "entrainment", "S_sub", sample.S_sub)
        setv(cfg, "entrainment", "f_ent", sample.f_ent)
        setv(cfg, "entrainment", "L0", sample.L0_m)
        setv(cfg, "lpt", "N0_cm3", sample.N0_cm3)
        setv(cfg, "lpt", "kappa_hygro", sample.kappa_hygro)
        setv(cfg, "lpt", "population_mode", args.population_mode)
        setv(cfg, "lpt", "nparticles", int(diag["nparticles"]))
        setv(cfg, "lpt", "particle_weight", float(diag["particle_weight"]))

        ini_path = cdir / f"{case}.ini"
        write_ini(cfg, ini_path)
        if args.make_netcdf:
            maybe_make_netcdf(cdir, case, cfg)

        manifest = {
            "case": case,
            "lhs_index": i,
            "lhs_seed": args.seed,
            "parameters": asdict(sample),
            "derived": diag,
            "definitions": {
                "tau_mix": "(L0^2/epsilon)^(1/3)",
                "tau_evap": "(r0^2-r_ccn^2)/(2*G*abs(S_sub))",
                "Da_nominal": "tau_mix/tau_evap",
                "tau_eta": "sqrt(nu_air/epsilon)",
                "St0": "tau_p(r0)/tau_eta; tau_p=2*rho_w*r^2/(9*rho_air*nu_air)",
            },
        }
        (cdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        row = {"case": case, **asdict(sample)}
        row.update({k: v for k, v in diag.items() if k != "warnings"})
        row["warnings"] = " | ".join(diag["warnings"])
        rows.append(row)

    csv_path = outroot / "case_table.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_slurm(outroot, args.n, args)

    metadata = {
        "n_cases": args.n,
        "seed": args.seed,
        "parameter_bounds": PARAM_BOUNDS,
        "epsilon_sampling": "uniform in log10(epsilon)",
        "other_sampling": "uniform in physical coordinate",
        "population_mode": args.population_mode,
        "fixed_np": args.fixed_np,
    }
    (outroot / "design.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.n} cases to {outroot}")
    print(f"Submit with: sbatch {outroot/'submit_array.slurm'}")


if __name__ == "__main__":
    main()
