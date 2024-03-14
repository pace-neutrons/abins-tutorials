from pathlib import Path

import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import ExpCellFilter, StrainFilter
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE, LBFGS, QuasiNewton
# from mace.calculators import mace_off
from mace.calculators import mace_mp
import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from rich import print as rprint
from tqdm import tqdm


DISP_SIZE = 1e-3  # Finite displacement  for phonon calculation
PHONOPY_FILE = "phonopy_params.yaml"  # Save file for force constants
SUPERCELL = [4, 4, 2]  # Supercell expansion for phonon calculation
SYMPREC = 1e-4  # Symmetry-detection distance threshold


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


def main():
    rprint(
        '[bold]Demonstration: MLIP + [italic]"direct method"[/italic] phonons + Incoherent INS[/bold]'
    )

    rprint("Step 0: set up the MLIP (MACE-MP-0)...")
    calc = mace_mp(model="medium", device="cpu", default_dtype="float64")

    rprint("Step 1: structure optimisation...")
    # atoms = ase.io.read("geometry.in")
    atoms = ase.io.read("opt.traj")
    atoms = get_optimized_geometry(atoms, calc)

    rprint("Step 2: set up phonon displacements...")
    # Phonopy uses its own ASE-like structure container
    phonopy = Phonopy(phonopy_from_ase(atoms), supercell_matrix=SUPERCELL, symprec=SYMPREC)
    phonopy.generate_displacements(distance=DISP_SIZE)

    rprint("Step 3: calculate forces on displacements...")
    force_container = []

    def _get_forces(atoms: Atoms) -> np.ndarray:
        atoms.calc = calc
        return atoms.get_forces()

    all_forces = [
        _get_forces(ase_from_phonopy(supercell))
        for supercell in tqdm(phonopy.supercells_with_displacements)
    ]

    rprint("Step 4: Construct force constants...")
    phonopy.forces = all_forces
    phonopy.produce_force_constants()

    rprint(f"      Saving to files {PHONOPY_FILE} and force_constants.hdf5 ...")
    phonopy.save(filename=PHONOPY_FILE, settings={"force_constants": False})
    from phonopy.file_IO import write_force_constants_to_hdf5
    write_force_constants_to_hdf5(phonopy.force_constants)

if __name__ == "__main__":
    main()
