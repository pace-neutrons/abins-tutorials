.. _mlip-phonopy:

Using a pre-trained MLIP with Phonopy
=====================================

For accurate results one would typically compute phonons with density-functional theory or with a system-specific interatomic potential. However, at the early stages of a project it may be useful to run a "general" machine-learned interatomic potential (MLIP) which is been trained on a variety of systems.

In this tutorial we primarily use the MACE-OFF23 [#mace-off]_ potential which is trained on organic molecules; the more general MACE-MP-0 [#mace-mp-0]_ can be used in the same way for other chemistries. In both cases we are using the ASE Calculator interface; most popular MLIP packages have an ASE interface and so can be used in a similar fashion.

We will also use ASE for geometry optimisation and Phonopy for setting-up/post-processing the force-constant calculations.

This example workflow is implemented as a small Python program, with
steps broken out into different functions. We will examine the
functions separately in this document: to test the overall workflow
you can run the program, or implement your own using the building
blocks shown.


Step 0: setting up the machine-learned interatomic potential (MLIP)
-------------------------------------------------------------------

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
-----------------------------


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

.. rubric:: References

.. [#mace-off]   \D. P. Kovács et al. "MACE-OFF23: Transferable Machine Learning Force Fields for Organic Molecules" https://arxiv.org/abs/2312.15211
.. [#mace-mp-0]   \I. Batatia et al. "A foundation model for atomistic materials chemistry" https://arxiv.org/abs/2401.00096
