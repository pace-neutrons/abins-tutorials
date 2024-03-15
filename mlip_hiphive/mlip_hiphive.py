#! /usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import ase.io
import numpy as np
import typer
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE, QuasiNewton
from ase.spacegroup.symmetrize import FixSymmetry
from ase.units import fs
from hiphive import ClusterSpace, ForceConstantPotential, StructureContainer
from hiphive.utilities import prepare_structures
from phonopy import Phonopy
from phonopy.file_IO import write_force_constants_to_hdf5
from phonopy.structure.atoms import PhonopyAtoms
from rich import print as rprint
from tqdm import tqdm
from trainstation import Optimizer
from typing_extensions import Annotated

# Geometry optimisation parameters
SYMPREC = 1e-4  # Symmetry-detection distance threshold

# MD parameters
TRAJ_FILE = "md.traj"

# Sampling parameters
SAMPLE_SIZE = 128
CLUSTER_CUTOFFS = [3.5, 2.5, 2.]


class MLIP(str, Enum):
    MACE_MP_0 = "mace-mp-0"
    MACE_OFF_23 = "mace-off"
    MACE_ANI_CC = "mace-ani-cc"


class Size(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"


class FloatSize(str, Enum):
    FLOAT32 = "32"
    FLOAT64 = "64"


def get_optimized_geometry(
    atoms: Atoms,
    calc: Calculator,
    trajectory_file: Path | str = "opt.traj",
    fire_steps: int = 30,
    qn_steps: int = 30,
    fmax: float = 1e-6,
    fix_symmetry: bool = True,
) -> Atoms:
    """Optimize geometry in two stages using ASE

    The FIRE optimiser is very stable, and can be a good starting point if we
    are starting far from the minimum-energy configuration. We then switch to
    the QuasiNewton optimiser which is more efficient when close to the minimum.

    Returns an optimised copy of the input atoms with calc attached.

    """

    atoms = atoms.copy()
    atoms.calc = calc
    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms, symprec=SYMPREC))

    traj = ase.io.Trajectory(trajectory_file, "w", atoms)

    opt = FIRE(atoms)
    opt.attach(traj)
    opt.run(steps=fire_steps)

    opt = QuasiNewton(atoms)
    opt.attach(traj)
    opt.run(fmax=fmax, steps=qn_steps)

    del atoms.constraints
    return atoms


def phonopy_from_ase(atoms: Atoms) -> PhonopyAtoms:
    """Convert ASE Atoms to PhonopyAtoms object"""
    return PhonopyAtoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        scaled_positions=atoms.get_scaled_positions(),
    )


def ase_from_phonopy(phonopy_atoms: PhonopyAtoms) -> Atoms:
    """Convert PhonopyAtoms to ASE Atoms object"""
    return Atoms(
        phonopy_atoms.symbols,
        cell=phonopy_atoms.cell,
        scaled_positions=phonopy_atoms.scaled_positions,
        pbc=True,
    )


def run_md(
    atoms: Atoms,
    pre_steps: int = 1000,
    run_steps: int = 2000,
    timestep_fs: float = 1.0,
    temperature_K=100,
    friction_fs: float = 0.01,
    traj_file: str | Path = TRAJ_FILE,
    traj_interval=10,
    n_snapshots=20,
    n_independent_runs=4,
) -> None:
    initial_positions = atoms.positions

    traj_writer = ase.io.Trajectory(traj_file, "w", atoms)

    for run in range(1, n_independent_runs + 1):
        rprint(f"[bold]MD run {run}/{n_independent_runs}[/bold]")

        atoms.positions = initial_positions
        MaxwellBoltzmannDistribution(atoms, temperature_K=(temperature_K * 2))

        dyn = Langevin(
            atoms,
            timestep=timestep_fs * fs,
            temperature_K=temperature_K,
            friction=(friction_fs / fs),
        )

        rprint("    Thermalising...")
        for _ in tqdm(dyn.irun(pre_steps), total=pre_steps):
            pass

        dyn.attach(traj_writer.write, interval=traj_interval)

        rprint(f"    Main run, writing to {traj_file}...")
        for _ in tqdm(dyn.irun(run_steps), total=run_steps):
            pass


def get_snapshots(traj_file: str | Path, n: int = SAMPLE_SIZE):
    traj = ase.io.read(traj_file, index=":")
    rng = np.random.default_rng()
    indices = rng.choice(len(traj), size=n, replace=False)

    rprint("  Indices from trajectory:")
    rprint("  {}".format(", ".join(map(str, sorted(indices)))))
    return [traj[i] for i in indices]


def save_fcp_to_phonopy(fcp: ForceConstantPotential,
                        supercell_matrix: np.ndarray,
                        atoms: Atoms,
                        fc_dir: Path = "force_constants") -> None:
    from os import makedirs
    makedirs(fc_dir, exist_ok=True)

    prim = fcp.primitive_structure
    phonopy = Phonopy(phonopy_from_ase(prim), supercell_matrix=supercell_matrix)
    supercell = ase_from_phonopy(phonopy.get_supercell())

    fcs = fcp.get_force_constants(supercell)
    phonopy.set_force_constants(fcs.get_fc_array(order=2))

    phonopy.save(
        settings={"force_constants": False}, filename=(fc_dir / "phonopy.yaml")
    )

    write_force_constants_to_hdf5(phonopy.force_constants, filename=(fc_dir / "force_constants.hdf5"))


def main(
    filename: Annotated[str, typer.Argument()] = "geometry.in",
    skip_opt: bool = False,
    md_traj: Optional[Path] = None,
    model: MLIP = MLIP.MACE_OFF_23,
    size: Size = Size.SMALL,
    float_size: FloatSize = FloatSize.FLOAT32,
    gpu: bool = False,
    supercell: Tuple[int, int, int] = (2, 2, 1),
    fc_dir: Path = "force_constants"
):
    device = "cuda" if gpu else "cpu"

    rprint("[bold]Demonstration: MLIP + sampled phonons + Incoherent INS[/bold]")

    rprint(f"Step 0: set up the MLIP ({model})...".format)

    match model:
        case MLIP.MACE_OFF_23:
            from mace.calculators import mace_off as mace_pretrained
            mace_kwargs = {"model": size}

        case MLIP.MACE_MP_0:
            from mace.calculators import mace_mp as mace_pretrained
            mace_kwargs = {"model": size}

        case MLIP.MACE_ANI_CC:
            from mace.calculators import mace_anicc as mace_pretrained

        case _:
            raise ValueError(f"Model '{model}' is not supported")

    calc = mace_pretrained(device=device, default_dtype=f"float{float_size}", **mace_kwargs)

    rprint("Step 1: structure optimisation with symmetry...")
    atoms = ase.io.read(filename)
    if (skip_opt or md_traj):
        rprint(":scissors: ---- skipped --- :scissors:")
    else:
        atoms = get_optimized_geometry(atoms, calc)

    rprint("Step 2: molecular dynamics...")

    md_atoms = atoms.copy() * supercell
    md_ref_cell = md_atoms.copy()
    md_atoms.calc = calc

    if md_traj:
        rprint(":scissors: ---- skipped --- :scissors:")
        traj_file = md_traj
    else:
        run_md(md_atoms, traj_file=TRAJ_FILE)
        traj_file = TRAJ_FILE

    rprint("Step 3: Randomly draw samples from trajectory")
    snapshots = get_snapshots(traj_file=traj_file)

    rprint("Step 4: Setup hiphive force constants container")

    cs = ClusterSpace(atoms.copy(), CLUSTER_CUTOFFS)
    sc = StructureContainer(cs)

    snapshots = prepare_structures(snapshots, md_ref_cell, check_permutation=False)

    for snapshot in snapshots:
        sc.add_structure(snapshot)

    rprint("Step 5: train force constants model")
    opt = Optimizer(sc.get_fit_data(), train_size=0.9)
    opt.train()

    print(opt.summary)
    fcp = ForceConstantPotential(cs, opt.parameters)

    rprint(f"... writing to {fc_dir}/phonopy.yaml & {fc_dir}/force_constants.hdf5")
    save_fcp_to_phonopy(fcp, supercell, atoms, fc_dir=fc_dir)    


if __name__ == "__main__":
    typer.run(main)
