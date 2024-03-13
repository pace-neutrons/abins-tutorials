import ase.io
from ase.constraints import ExpCellFilter, StrainFilter
from ase.optimize import QuasiNewton, FIRE, LBFGS
from mace.calculators import mace_off
from rich import print as rprint

def main():
    rprint("[bold]Demonstration: MLIP + [italic]\"direct method\"[/italic] phonons + Incoherent INS[/bold]")

    rprint("Step 0: set up the MLIP (MACE-OFF23)...")
    calc = mace_off(model="medium")
    
    rprint("Step 1: structure optimisation...")
    atoms = ase.io.read("geometry.in")
    atoms.calc = calc

    rprint("    Fixed unit-cell")
    opt = FIRE(atoms)
    opt.run(steps=30)

    rprint("    Fixed atoms")
    opt = LBFGS(StrainFilter(atoms))
    opt.run(steps=20)

    rprint("    Full optimization")
    opt = QuasiNewton(ExpCellFilter(atoms))
    opt.run(fmax=1e-3)

    rprint(atoms)

if __name__ == '__main__':
    main()
