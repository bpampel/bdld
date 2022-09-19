.. _fes:

Free energy surface of system [fes]
********************************************

Estimate the free energy surface from the :ref:`[histogram] action <histogram>`

This uses the given temperature to calculate the free energy of each histogram bin.
The output is aligned to have the minimum at zero.

It can also plot the FES directly via matplotlib.

**stride**: *int*
  Time steps between FES calculations.

**kt**: *float*
  Thermal energy of the system, should usually be the same as the one specified in the [ld] section

**filename**: *string*, optional
  Filename to write the FES to.

**write-stride**: *int*, optional
  Write FES to file every n time steps (filename will be appended by :code:`_t` (with the current time step t).
  If not specified the FES will only be written once at the end of the simulation.

**fmt**: *string*, defaults to "%14.9"
  c-style format specifier used when writing data to file

**plot-filename**: *string*, optional
  If specified, the FES will be plotted and saved to the specified path.
  This works only for 1D and 2D.

**plot-stride**: *int*, optional
  Plot FES to file every n time steps, not only at the end of the simulation

**plot-domain**: *list of floats*, optional
  Range of free energy values that should be shown in plot

Example
^^^^^^^

Calculate FES every 100 steps, save it to file every 500 steps and save a plot at the end of the simulation:
::

  [fes]
  stride: 100
  write-stride: 500
  filename: fes
  plot-filename: fes.png
  plot-domain: 0, 8.5
