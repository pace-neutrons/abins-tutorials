#! /usr/bin/env python3

from enum import Enum
from pathlib import Path
from typing import Literal, Tuple
from typing_extensions import Annotated

import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import ExpCellFilter, StrainFilter
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE, LBFGS, QuasiNewton
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from rich import print as rprint
from tqdm import tqdm
import typer


DISP_SIZE = 1e-3  # Finite displacement  for phonon calculation
PHONOPY_FILE = "phonopy_params.yaml"  # Save file for force constants
SYMPREC = 1e-4  # Symmetry-detection distance threshold


class MLIP(str, Enum):
    MACE_MP_0 = "mace-mp-0"
    MACE_OFF_23 = "mace-off"


def get_optimized_geometry(
    atoms: Atoms,
    calc: Calculator,
    trajectory_file: Path | str = "opt.traj",
    fire_steps=30,
    fmax=1e-6,
) -> Atoms:
    """Optimize geometry in two stages using ASE

    The FIRE optimiser is very stable, and can be a good starting point if we
    are starting far from the minimum-energy configuration. We then switch to
    the QuasiNewton optimiser which is more efficient when close to the minimum.

    Returns an optimised copy of the input atoms with calc attached.

    """

    atoms = atoms.copy()
    atoms.calc = calc

    traj = Trajectory(trajectory_file, "w", atoms)

    opt = FIRE(atoms)
    opt.attach(traj)
    opt.run(steps=fire_steps)

    opt = QuasiNewton(atoms)
    opt.attach(traj)
    opt.run(fmax=fmax)

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


def main(
    filename: Annotated[str, typer.Argument()] = "geometry.in",
    model: MLIP = MLIP.MACE_OFF_23,
    gpu: bool = False,
    supercell: Tuple[int, int, int] = (4, 4, 2),
):
    device = "cuda" if gpu else "cpu"

    rprint(
        '[bold]Demonstration: MLIP + [italic]"direct method"[/italic] phonons + Incoherent INS[/bold]'
    )

    rprint(f"Step 0: set up the MLIP ({model})...".format)

    if model == MLIP.MACE_OFF_23:

        from mace.calculators import mace_off

        calc = mace_off(model="medium", device=device, default_dtype="float64")

    elif model == MLIP.MACE_MP_0:

        from mace.calculators import mace_mp

        calc = mace_mp(model="medium", device=device, default_dtype="float64")

    else:
        raise ValueError(f"Model '{model}' is not supported")

    rprint("Step 1: structure optimisation...")
    atoms = ase.io.read(filename)
    atoms = get_optimized_geometry(atoms, calc)

    rprint("Step 2: set up phonon displacements...")
    # Phonopy uses its own ASE-like structure container
    phonopy = Phonopy(
        phonopy_from_ase(atoms), supercell_matrix=supercell, symprec=SYMPREC
    )
    phonopy.generate_displacements(distance=DISP_SIZE)

    rprint("Step 3: calculate forces on displacements...")
    force_container = []

    def _get_forces(atoms: Atoms) -> np.ndarray:
        atoms.calc = calc
        return atoms.get_forces()

    all_forces = [
        _get_forces(ase_from_phonopy(displaced_supercell))
        for displaced_supercell in tqdm(phonopy.supercells_with_displacements)
    ]

    rprint("Step 4: Construct force constants...")
    phonopy.forces = all_forces
    phonopy.produce_force_constants()

    rprint(f"      Saving to files {PHONOPY_FILE} and force_constants.hdf5 ...")
    phonopy.save(filename=PHONOPY_FILE, settings={"force_constants": False})
    from phonopy.file_IO import write_force_constants_to_hdf5

    write_force_constants_to_hdf5(phonopy.force_constants)


if __name__ == "__main__":
    typer.run(main)
