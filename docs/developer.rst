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
class, and have to implement at least the `run()` and `final_run()` methods to
be useable in the main loop.

The core of this code is surely the `BirthDeath <source/bdld.actions.html#module-bdld.actions.birth_death>`_
action, see there for how the algorithm is implemented.
