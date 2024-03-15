Using a pre-trained MLIP with Phonopy
=====================================

For accurate results one would typically compute phonons with density-functional theory or with a system-specific interatomic potential. However, at the early stages of a project it may be useful to run a "general" machine-learned interatomic potential (MLIP) which is been trained on a variety of systems.

In this tutorial we primarily use the MACE-OFF23 potential which is trained on organic molecules; the more general MACE-MP-0 can be used in the same way for other chemistries. In both cases we are using the ASE Calculator interface; most popular MLIP packages have an ASE interface and so can be used in a similar fashion.

We will also use ASE for geometry optimisation and Phonopy for setting-up/post-processing the force-constant calculations.

