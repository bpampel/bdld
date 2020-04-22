#!/usr/bin/env python3

import argparse
import numpy as np
from potential import Potential


def parse_cliargs():
    """Use argparse to get cli arguments
    :return: args: Namespace with cli arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-kT', '--temp', type=float, dest='kT',
                        help="Energy (in units of kT) of the FES file", required=True)
    parser.add_argument('-g', '--friction', type=float,
                        help="Friction coefficient", required=True)
    parser.add_argument('-dt', '--time-step', type=float, dest='time_step',
                        help="Time step", required=True)
    parser.add_argument('--num-steps', type=int, dest='num_steps',
                        help="Time step", required=True)
    parser.add_argument('--random-seed', type=int, dest='seed',
                        help="Seed for the random number generator")
    parser.add_argument('--initial-pos', type=float, dest='initial_pos', nargs='+',
                        required=True, help="Initial position of particle.")
    return parser.parse_args()


def main():
    # custom stuff for testing
    print_freq = 100
    # exemplary potentials for testing, move to input
    double_well = np.array([0, 0.7, -4, 0, 1])
    wolfe_quapp = np.array([[ 0. ,  0.1, -4. ,  0. ,  1. ],
                            [ 0.3,  1. ,  0. ,  0. ,  0. ],
                            [-2. ,  0. ,  0. ,  0. ,  0. ],
                            [ 0. ,  0. ,  0. ,  0. ,  0. ],
                            [ 1. ,  0. ,  0. ,  0. ,  0. ]])


    args = parse_cliargs()
    dt = args.time_step

    rng = np.random.default_rng(args.seed)

    mass = 1.0  # make variable?
    one_over_mass = 1. / mass

    # coefficients for thermostat
    c1 = np.exp(-0.5 * args.friction * args.kT)
    c2 = np.sqrt((1 - c1 * c1) * mass * args.kT)


    pot = Potential(wolfe_quapp)


    # initialization
    dim = pot.dimension
    if len(args.initial_pos) != dim:
        raise ValueError("Intial position has not the same dimensions as potential (required: {})"
                         .format(dim))
    pos = args.initial_pos
    mom = np.array([0.0] * dim)
    energy, forces = pot.evaluate(pos)
    rand_gauss = rng.standard_normal(dim)

    print("i: position, mom, energy")
    print("0: {}, {}, {}".format(pos,mom,energy))


    # velocity verlet with bussi-parinello thermostat
    for i in range(1, 1 + args.num_steps):
        # first part of thermostat
        mom = c1 * mom + c2 * rand_gauss
        # velocity verlet
        pos += mom * one_over_mass * dt + 0.5 * forces * one_over_mass * dt
        energy, new_forces = pot.evaluate(pos)
        mom += 0.5 * (forces + new_forces) * dt
        # second part of thermostat
        rand_gauss = rng.standard_normal(dim)
        mom = c1 * mom + c2 * rand_gauss
        if i % print_freq == 0:
            print("{}: {}, {}, {}".format(i,pos,mom,energy))
        forces = new_forces


if __name__ == '__main__':
    main()
