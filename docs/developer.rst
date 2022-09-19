.. _developer:

Information for developers
**************************

This provides some short overview on the code and the ideas and principles being used.

Everything was written in python3.6, see the `setup.py` file for the versions of
numpy and scipy that the code was designed with.

The code uses classes and inheritance and generally follows more the OOP approach.

Also the code attempts to use the
`typing <https://docs.python.org/3/library/typing.html>`_ module for annotations
wherever possible, although this is currently throwing some warnings and errors and
not fully carried out.

For installation and distribution this uses a simple Makefile and the setup module.
The documentation is done in rst format and created with Sphinx.


Short explanation of the program flow
=====================================
The central unit is the `main.py` file, it creates all the actions and runs the
main loop over the time steps.

After reading the cli arguments (containing the input file) it then calls the
`inputparser.py` file.
This defines some custom methods on top of the `configparser <https://docs.python.org/3/library/configparser.html>`_ module (e.g. to allow
lists) and then defines all possible input options and does the input parsing.

The resulting options are then processed in the `main.py` files to set up all
*actions*, sadly some of the code is duplicated here currently (the
`inputparser.py` options have to be passed as the arguments of the respective
setup functions of the actions)

After setup, the main loop is called for the desired number of iterations.
The `run()` method of all actions is called in a fixed order in the loop,
and not in the order they appear in the file.
After the loop has finished the `final_run()` method of all actions is called to
allow for final clean up and saving operations.


Actions
=======

Everything that happens after initialization is an *action*, e.g. the
Langevin dynamics, the birth-death steps but also the data storage
and analysis (e.g. calculating and storing the FES).

All actions are subclasses of the `bdld.action.Action <source/bdld.actions.html#module-bdld.actions.action>`_
class, and have to implement at least the `run()` method to be useable in the main
loop.

The core of this code is surely the `BirthDeath <source/bdld.actions.html#module-bdld.actions.birth_death>`_
action, see there for how the algorithm is implemented.



Currently under development / Upcoming
======================================

The `bd_estimate_FES` branch hosts first attempts of using the *estimated FES* for the birth-death term (instead of the exact one from the potential which is typically not available).

While the basic idea is implemented, this is still work-in-progress to allow for more sophisticated schemes.

Several problems have to be addressed:

 - The main branch only allows to have each action type once, so the same histogram/FES has to be used for the birth-death scheme as for the sampling analysis. The code therefore rewrites parts of the `main.py` and `inputparser.py` to allow multiple actions of the same type.
 - We want to be able to have more options on how to determine/set the equilibrium density (similarly to a "target distribution"). As a first idea, I plan to implement a "uniform density" to test the behavior, as well as estimating it from a specific histogram.
 - More sophisticated histogram schemes could then be implemented (e.g. as simple idea: introduce decay)
