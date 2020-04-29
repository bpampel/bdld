#!/usr/bin/env python3

import argparse
import numpy as np
from bussi_parinello_md import BussiParinelloMD as bpmd
from birth_death import BirthDeath
from potential import Potential


def parse_cliargs():
    """Use argparse to get cli arguments
    :return: args: Namespace with cli arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-kT', '--temp', type=float, dest='kt',
                        help="Energy (in units of kT) of the FES file", required=True)
    parser.add_argument('-g', '--friction', type=float,
                        help="Friction coefficient", required=True)
    parser.add_argument('-dt', '--time-step', type=float, dest='time_step',
                        help="Time step", required=True)
    parser.add_argument('--num-steps', type=int, dest='num_steps',
                        help="Time step", required=True)
    parser.add_argument('-bw', '--kernel-bandwidth', type=float, dest='bw',
                        help="Bandwidth for gaussian kernels", required=True)
    parser.add_argument('--birth-death-stride', type=int, dest='bd_stride',
                        help="Stride for the birth-death processes", required=True)
    parser.add_argument('--random-seed', type=int, dest='seed',
                        help="Seed for the random number generator")
    parser.add_argument('--initial-pos', type=float, dest='initial_pos', nargs='+',
                        required=True, help="Initial position of particle.")
    return parser.parse_args()


def main():
    # custom stuff for testing
    print_freq = 100
    # exemplary potentials for testing, move to input in the end
    double_well = np.array([0, 0.7, -4, 0, 1])
    wolfe_quapp = np.array([[ 0. ,  0.1, -4. ,  0. ,  1. ],
                            [ 0.3,  1. ,  0. ,  0. ,  0. ],
                            [-2. ,  0. ,  0. ,  0. ,  0. ],
                            [ 0. ,  0. ,  0. ,  0. ,  0. ],
                            [ 1. ,  0. ,  0. ,  0. ,  0. ]])

    args = parse_cliargs()

    # set up MD and BD
    md = bpmd(Potential(wolfe_quapp),
              args.time_step,
              args.friction,
              args.kt,
              args.seed,
              )
    if args.seed is not None:
        args.seed += 1000
    bd = BirthDeath(md.particles, args.time_step, args.bw, args.seed)


    # testing again
    for _ in range(5):
        md.add_particle(args.initial_pos)
    p = md.particles[2]  # alias for test logging
    print("i: position, mom, energy")
    print("0: {}, {}, {}".format(p.pos, p.mom, p.energy))

    # run MD
    for i in range(1, 1 + args.num_steps):
        md.step()
        bd.step()
        if i % print_freq == 0:
            print("{}: {}, {}, {}".format(i, p.pos, p.mom, p.energy))

if __name__ == '__main__':
    main()
