#!/usr/bin/env python3
"""Create the minimal MicroHH <case>_input.nc required for this passive-scalar box.

The actual 3-D dry filament/blob is imposed by the C++ entrainment initializer;
this NetCDF only supplies full-level z and zero mean profiles required by MicroHH.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import configparser
import numpy as np


def read_cfg(path: Path):
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.optionxform = str
    cfg.read(path)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ini", required=True, type=Path)
    p.add_argument("--case", required=True)
    p.add_argument("--outdir", type=Path, default=Path("."))
    args = p.parse_args()

    try:
        import netCDF4 as nc4
    except ImportError as exc:
        raise SystemExit("python-netCDF4 is required to generate MicroHH input files") from exc

    cfg = read_cfg(args.ini)
    nz = cfg.getint("grid", "ktot")
    lz = cfg.getfloat("grid", "zsize")
    dz = lz / nz
    z = (np.arange(nz, dtype=np.float64) + 0.5) * dz

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{args.case}_input.nc"
    if out.exists():
        out.unlink()

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

    print(out)


if __name__ == "__main__":
    main()
