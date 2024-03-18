.. _mlip-phonopy:

=======================================
 Using a pre-trained MLIP with Phonopy
=======================================

For accurate results one would typically compute phonons with density-functional theory or with a system-specific interatomic potential. However, at the early stages of a project it may be useful to run a "general" machine-learned interatomic potential (MLIP) which is been trained on a variety of systems.

In this tutorial we primarily use the MACE-OFF23 [#mace-off]_ potential which is trained on organic molecules; the more general MACE-MP-0 [#mace-mp-0]_ can be used in the same way for other chemistries. In both cases we are using the ASE Calculator interface; most popular MLIP packages have an ASE interface and so can be used in a similar fashion.

We will also use ASE for geometry optimisation and Phonopy for setting-up/post-processing the force-constant calculations.

This example workflow is implemented as a small Python program, with
steps broken out into different functions. We will examine the
functions separately in this document: to test the overall workflow
you can run the program, or implement your own using the building
blocks shown.


Step 0: setting up the machine-learned interatomic potential (MLIP)
===================================================================

First, we need to set up an MLIP implementation. Here we assume that
the MACE package has already been installed into the current
environment. (See: :doc:`environment-setup`.) Then we can create an
ASE calculator, ready to attach to a structure (Atoms) object.


.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
   :start-after:  if model == MLIP.MACE_OFF_23:
   :end-before:   elif model == MLIP.MACE_MP_0:

There are a range of MLIP packages available and most of them provide
an ASE calculator, or have had one wrapped around them. Here are a
few, in no particular order:

=========================================================================================   ============================================
MLIP package                                                                                ASE interface
=========================================================================================   ============================================
`QUIP (GAP) <https://libatoms.github.io/GAP>`_                                              `quippy <https://libatoms.github.io/GAP/quippy-potential-tutorial.html>`_
`JuLIP <https://github.com/JuliaMolSim/JuLIP.jl>`_ `(ACE) <https://acesuit.github.io>`_     `pyjulip <https://github.com/casv2/pyjulip>`_
`MACE <https://mace-docs.readthedocs.io>`_                                                  Yes
`pacemaker (ACE) <https://pacemaker.readthedocs.io>`_                                       Yes
`GPUMD (NEP) <https://gpumd.org>`_                                                          `calorine <https://calorine.materialsmodeling.org>`_
`NequIP <https://github.com/mir-group/nequip>`_                                             Yes
`n2p2 <https://github.com/CompPhysVienna/n2p2>`_                                            via LAMMPS
=========================================================================================   ============================================


Step 1: geometry optimisation
=============================


First, we grab an input structure for hexane from the CCSD in CIF
format. The file provided by CCSD has already been converted to a
simpler format and saved as a "geometry.in" file in the FHI-aims
format. (As we will be using ASE we can work with whichever format
looks the nicest!)

We then optimise the geometry with a very fine force criterion, in
order to get as close to the minimum-energy configuration as
possible.

.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
    :start-at: atoms = ase.io.read(filename)
    :end-at: atoms = get_optimized_geometry(atoms, calc)

.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
   :start-at: def get_optimized_geometry
   :end-at:   return atoms

To get a sense of the optimisation progress, examine the trajectory file with

.. code-block:: sh

   ase gui opt.traj

and create a plot of max forces by entering ``i, fmax`` in the Graphs window.
   
.. image:: Figures/geomopt-screenshot.png
           :alt: Sreenshot of ASE-gui showing hexane structure with plots of energy and force convergence

Step 2: finite displacements
============================

We will use the `Phonopy <https://phonopy.github.io/phonopy/index.html>`_
Python API to create supercells with displaced atoms and analyse the forces.

A few lines of code are needed to adapt between the ASE and phonopy structure representations: to keep things tidy we wrap these into functions.

.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
                    :start-at: def phonopy_from_ase(atoms: Atoms)
                    :end-before: def main(

Now we set up the displacements: the displaced structures are created on the Phonopy object as ``phonopy.supercells_with_displacements``. SYMPREC (symmetry threshold) and DISP_SIZE (finite displacement distance) are parameters controlling the displacement scheme; in the sample program they are set to 1e-4 and 1e-3 respectively.
The supercell matrix is another user parameter; this has a significant impact on the runtime and the quality of results, so should be checked carefully.

.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
   :start-after: rprint("Step 2: set up phonon displacements...")
   :end-before:  rprint("Step 3: calculate forces on displacements...")

Step 3: force calculations
==========================

Using ASE the idiom for calculating forces on a structure is:

- attach a "calculator" to the structure ``atoms.calc = calculator``
- call ``atoms.get_forces())``

Here we do this for each of the displaced supercells, collecting the results into a list for later use.
`tqdm <https://tqdm.github.io>`_ is used to make a nice progress bar while this runs; for large supercells it may take a while!
  
.. literalinclude:: ../mlip_phonopy/mlip_phonopy.py
   :start-after: rprint("Step 3: calculate forces on displacements...")
   :end-before:  rprint("Step 4: Construct force constants...")
                               
.. rubric:: References

.. [#mace-off]   \D. P. Kovács et al. "MACE-OFF23: Transferable Machine Learning Force Fields for Organic Molecules" https://arxiv.org/abs/2312.15211
.. [#mace-mp-0]   \I. Batatia et al. "A foundation model for atomistic materials chemistry" https://arxiv.org/abs/2401.00096
