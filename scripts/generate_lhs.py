#!/usr/bin/env python3
from __future__ import annotations
import argparse, configparser, csv, json, math, shutil
from pathlib import Path
import numpy as np
from scipy.stats import qmc

RANGES = {
    "epsilon": (-4.0, -2.0),       # sampled uniformly in log10
    "S_sub": (-0.50, -0.05),
    "N0_cm3": (50.0, 500.0),
    "f_ent": (0.10, 0.50),
    "L0_m": (0.05, 0.50),
    "kappa": (0.01, 0.60),
}

NU = 1.5e-5
KAPPA_T = 2.0e-5
DOMAIN_VOLUME_M3 = 1.0
RHO_W = 997.0
MU_AIR = 1.8e-5
R0_WET = 10e-6
NP_FIXED = 12_500_000


def lhs_samples(n: int, seed: int):
    sampler = qmc.LatinHypercube(d=6, seed=seed, scramble=True)
    u = sampler.random(n)
    lo = np.array([RANGES[k][0] for k in RANGES])
    hi = np.array([RANGES[k][1] for k in RANGES])
    x = qmc.scale(u, lo, hi)
    out=[]
    for row in x:
        logeps,Ssub,N0,fent,L0,kappa=row
        out.append(dict(epsilon=10.0**logeps,S_sub=Ssub,N0_cm3=N0,
                        f_ent=fent,L0_m=L0,kappa=kappa))
    return out


def derived(p):
    eps=p["epsilon"]
    eta=(NU**3/eps)**0.25
    tau_eta=(NU/eps)**0.5
    # Stokes relaxation time for a representative 10 um water droplet.
    tau_p=2.0*RHO_W*R0_WET**2/(9.0*MU_AIR)
    St_eta=tau_p/tau_eta
    # Large-eddy turnover time based on inertial scaling.
    tau_mix=(p["L0_m"]**2/eps)**(1.0/3.0)
    # Nominal evaporation time uses |S_sub| and constant G from base template.
    G=1e-10
    tau_evap=max((R0_WET**2-(50e-9)**2)/(2.0*G*abs(p["S_sub"])),1e-12)
    Da_L0=tau_mix/tau_evap
    dx=1.0/256.0
    N_phys=p["N0_cm3"]*1e6*DOMAIN_VOLUME_M3
    weight=N_phys/NP_FIXED
    return dict(eta_m=eta,dx_over_eta=dx/eta,tau_eta_s=tau_eta,
                tau_p_s=tau_p,St_eta=St_eta,tau_mix_s=tau_mix,
                tau_evap_s=tau_evap,Da_L0=Da_L0,N_phys=N_phys,
                superparticle_weight=weight)


def read_ini(path: Path):
    cp=configparser.ConfigParser(interpolation=None)
    cp.optionxform=str
    cp.read(path)
    return cp


def write_case_ini(base_path: Path, out_path: Path, p, mode: str):
    cp=read_ini(base_path)
    cp["forcing"]["epsilon_target"]=f'{p["epsilon"]:.10e}'
    cp["entrainment"]["S_sub"]=f'{p["S_sub"]:.10e}'
    cp["entrainment"]["f_ent"]=f'{p["f_ent"]:.10e}'
    cp["entrainment"]["L0"]=f'{p["L0_m"]:.10e}'
    cp["lpt"]["N0_cm3"]=f'{p["N0_cm3"]:.10e}'
    cp["lpt"]["kappa_ccn"]=f'{p["kappa"]:.10e}'
    N_phys=round(p["N0_cm3"]*1e6*DOMAIN_VOLUME_M3)
    if mode=="strict_one_to_one":
        cp["lpt"]["population_mode"]="strict_one_to_one"
        cp["lpt"]["np"]=str(N_phys)
    else:
        cp["lpt"]["population_mode"]="weighted_fixed_np"
        cp["lpt"]["np"]=str(NP_FIXED)
    with out_path.open("w") as f: cp.write(f)


def write_slurm(root: Path, n: int, exe: str, partition: str, gpus: int, walltime: str):
    txt=f'''#!/bin/bash
#SBATCH --job-name=microhh_lpt_lhs
#SBATCH --array=0-{n-1}%10
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus={gpus}
#SBATCH --time={walltime}
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
CASE=$(printf "case_%03d" "$SLURM_ARRAY_TASK_ID")
cd "$ROOT/$CASE"

# Adjust module names for your site.
module purge || true
# module load cuda netcdf fftw boost cmake

EXE="{exe}"
$EXE init case
$EXE run case
'''
    (root/"submit_array.slurm").write_text(txt)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("lhs_cases"))
    ap.add_argument("-n","--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--population-mode", choices=["weighted_fixed_np","strict_one_to_one"], default="weighted_fixed_np")
    ap.add_argument("--microhh", default="../../build/microhh")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--walltime", default="08:00:00")
    args=ap.parse_args()

    args.out.mkdir(parents=True,exist_ok=True)
    samples=lhs_samples(args.samples,args.seed)
    rows=[]
    for i,p in enumerate(samples):
        d=derived(p); row={"case":i,**p,**d}; rows.append(row)
        case=args.out/f"case_{i:03d}"; case.mkdir(exist_ok=True)
        write_case_ini(args.base,case/"case.ini",p,args.population_mode)
        warnings=[]
        if d["dx_over_eta"]>2.0:
            warnings.append(f'DNS resolution warning: dx/eta={d["dx_over_eta"]:.2f} > 2')
        if args.population_mode=="strict_one_to_one" and d["N_phys"]>5e7:
            warnings.append("Very large strict 1:1 particle count; check GPU memory before submission.")
        manifest={**row,"population_mode":args.population_mode,"warnings":warnings}
        (case/"manifest.json").write_text(json.dumps(manifest,indent=2))

    with (args.out/"lhs_matrix.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    (args.out/"lhs_matrix.json").write_text(json.dumps(rows,indent=2))
    write_slurm(args.out,args.samples,args.microhh,args.partition,args.gpus,args.walltime)
    print(f"Wrote {args.samples} cases to {args.out}")

if __name__=="__main__": main()
