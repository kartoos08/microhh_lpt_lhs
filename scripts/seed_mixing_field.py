#!/usr/bin/env python3
"""Generate a deterministic, tri-periodic entrainment mask and write MicroHH's
initial passive-scalar binary field s.0000000.

The mask is generated from a Gaussian random field filtered around k0=2*pi/L0,
then thresholded by rank so the dry-air volume fraction is exactly f_ent up to
one grid cell. The cloudy region is assigned S_cloud and dry region S_sub.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def periodic_correlated_field(shape, lengths, L0, seed):
    nz, ny, nx = shape
    Lz, Ly, Lx = lengths
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(shape)
    q = np.fft.rfftn(white, axes=(0,1,2))

    kz = 2*np.pi*np.fft.fftfreq(nz, d=Lz/nz)[:, None, None]
    ky = 2*np.pi*np.fft.fftfreq(ny, d=Ly/ny)[None, :, None]
    kx = 2*np.pi*np.fft.rfftfreq(nx, d=Lx/nx)[None, None, :]
    kmag = np.sqrt(kx*kx + ky*ky + kz*kz)
    k0 = 2*np.pi/max(L0, np.finfo(float).tiny)
    # Broad band-pass centered at the requested eddy scale; suppress k=0.
    sigma_k = 0.55*k0
    filt = np.exp(-0.5*((kmag-k0)/sigma_k)**2)
    filt[0, 0, 0] = 0.0
    q *= filt
    phi = np.fft.irfftn(q, s=shape, axes=(0,1,2)).real
    phi -= phi.mean()
    std = phi.std()
    if std == 0:
        raise RuntimeError("Filtered random field has zero variance")
    return phi/std


def exact_fraction_mask(phi, fraction):
    flat = phi.ravel()
    n_dry = int(round(fraction * flat.size))
    n_dry = min(max(n_dry, 0), flat.size)
    mask = np.zeros(flat.size, dtype=bool)
    if n_dry:
        # Largest values define the dry intrusion; argpartition is O(N).
        idx = np.argpartition(flat, flat.size-n_dry)[flat.size-n_dry:]
        mask[idx] = True
    return mask.reshape(phi.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", required=True)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=1.0)
    p.add_argument("--lz", type=float, default=1.0)
    p.add_argument("--L0", type=float, required=True)
    p.add_argument("--f-ent", type=float, required=True)
    p.add_argument("--S-sub", type=float, required=True)
    p.add_argument("--S-cloud", type=float, default=0.0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", choices=("float32","float64"), default="float64")
    args = p.parse_args()

    case_dir = Path(args.case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    phi = periodic_correlated_field(
        (args.nz, args.ny, args.nx), (args.lz, args.ly, args.lx), args.L0, args.seed)
    dry = exact_fraction_mask(phi, args.f_ent)
    s = np.where(dry, args.S_sub, args.S_cloud).astype(args.dtype, copy=False)

    # MicroHH binary fields are laid out (k,j,i), i fastest in C order.
    out = case_dir / "s.0000000"
    s.tofile(out)
    np.savez_compressed(case_dir / "entrainment_mask_meta.npz",
                        dry_fraction=float(dry.mean()), L0=args.L0,
                        S_sub=args.S_sub, S_cloud=args.S_cloud,
                        seed=args.seed)
    print(f"wrote {out} ({s.nbytes/2**20:.1f} MiB), dry fraction={dry.mean():.8f}")

if __name__ == "__main__":
    main()
