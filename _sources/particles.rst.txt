.. _particles:

Particles / walkers [particles]
*******************************

Specify the number of particles and their initial distribution


**number**: *int*
  total number of particles

**initial-distribution**: *string*
  method to set initial positions of particles. Currently available values:

  * :code:`random-global`: initialize each particle randomly in the potential area
  * :code:`random-pos`: initialize each particle randomly in one of the given positions
  * :code:`fractions-pos`: initialize particles with fixed fractions at given positions

For *-pos* options you need to specify the wanted positions with :code:`pos1` to :code:`posN`:

**pos1**: *float* or *list of floats*
  Position in potential (list for more than one dimension)

For :code:`fractions-pos` also the fractions have to be given.

**fractions**: *list of float*
  Fractions of walkers that should be initalized at the specified positions.
  Needs to have one value for each position.
  The sum is normalized to one, so also absolute values can be specified.
  If the particles are not exactly distributable according to the fractions, the first position will get the remainder.

**mass**: *float*, optional
  Mass of the particles, defaults to 1.0

**seed**: *int*, optional
  Starting seed for the random number generator that is used if particles are distributed randomly


Example
^^^^^^^

This is an example where 100 particles are distributed on 3 positions of a 2D potential
::

  [particles]
  number: 100
  initial-distribution: fractions-pos
  pos1: 0.5, 0.5
  pos2: 1.0, -0.1
  pos3: -0.5, 1.1
  fractions: 0.3, 0.4, 0.3
