.. _trajectories:

Store particle positions [trajectories]
***************************************

Action to store and optionally write the position of the particles at specified intervals.

Just adding the section without any options will already store the trajectories temporarily in memory and make them available for histogramming.
All further options are thus only needed if the trajectories should be written to file.

**filename**: *string*, optional
  Filename to write the trajectories to. Will be extended by :code:`.i` for each particle (with i being the particle number)

**stride**: *int*, defaults to 1
  Time steps between saved positions of particles

**write-stride**: *int*, defaults to 100
  How often the writing to file should take place.
  This directly influences how many positions have to be held in memory during the simulation (:code:`write-stride * particle_number * potential_dimensions`)

**fmt**: *string*, defaults to "%14.9"
  c-style format specifier used when writing data to file


Example
^^^^^^^

Just store positions in memory for histogramming, but don't write them to file:
::

  [trajectories]

Write every 5th position (of all walkers) to files called `traj.1`, `traj.2`, etc:
::

  [trajectories]
  filename: traj
  stride: 5
