.. _environment-setup:

Environment setup
=================

The simplest way to get an appropriate environment set up for these
tutorials is using a conda environment.

If you are an ISIS facility users, a fast "mamba"-enable conda
impelementation is available on IDAaaS. Otherwise, if you are not sure
what is available it may be best to use `Miniforge <https://github.com/conda-forge/miniforge>`_ or `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_ to get up and running.

An environment file is provided in this repository; create an environment and activate it with e.g.

.. code-block:: sh

  mamba env create -f env.yaml
  mamba activate abins-tutorials

Be aware that this environment has about 3GB of depenencies; this is
dominated by the NVIDIA/CUDA libraries needed to run on GPU. If you
are not using GPU, consider removing the "pytorch-cuda" line before
installing!
