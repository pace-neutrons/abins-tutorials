.. abins-tutorials documentation master file, created by
   sphinx-quickstart on Fri Mar 15 15:46:57 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AbINS Tutorials!
===========================

This site collects some tutorials for advanced users of the AbINS
programs for simulation of inelastic neutron scattering phonon
spectra.

The code is available from
https://github.com/pace-neutrons/abins-tutorials : the "source" folder
contains this documentation, while the other folders contain
demonstration programs and some reference data.

The demonstration programs are designed to be somewhat useful
interactively: try running them with the "\-\-help" option
(e.g. ``mlip_phonopy/mlip_phonopy.py --help``) to see available
options. You should be able to point them at your own structure files
and have them do something useful --- but the more important goal is to
demonstrate principles and code idioms highlighted in the tutorials.

Other documentation
-------------------

The basic documentation for the AbINS algorithms is found on the
Mantid website:

- `Abins Algorithm <https://docs.mantidproject.org/nightly/algorithms/Abins-v1.html>`_
- `Abins2D Algorithm <https://docs.mantidproject.org/nightly/algorithms/Abins2D-v1.html>`_

with some further details in the "concepts" section

- `Structure factor calculation method <https://docs.mantidproject.org/nightly/concepts/DynamicalStructureFactorFromAbInitio.html>`_
- `Development notes <https://docs.mantidproject.org/nightly/concepts/AbinsImplementation.html>`_
- `Fast approximate broadening algorithm <https://docs.mantidproject.org/nightly/concepts/AbinsInterpolatedBroadening.html>`_

The academic reference for AbINS is the research paper in Physica B. [#Dymkowski2018]_

.. [#Dymkowski2018]   \K. Dymkowski et al. (2018) "AbINS: The modern software for INS interpretation" *Physica B: Cond. Matter* **551** 443-448

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   environment-setup.rst
   mlip-phonopy.rst
