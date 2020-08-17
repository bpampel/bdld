#!/usr/bin/env python3

import argparse
import numpy as np
import analysis
from bdld import BirthDeathLangevinDynamics
from bussi_parinello_ld import BussiParinelloLD as bpld
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
    parser.add_argument('-nw', '--num-walkers', type=int, dest='num_walkers', default=50,
                        help="Number of walkers to put in each minimum, defaults to 50")
    parser.add_argument('-w1', '--frac-walkers-first-state', type=float, dest='walkers_frac', default=0.5,
                        help="Fraction of walkers to put in first minimum "
                             "(1 is all in first, 0 is all in second, will be rounded in favor of first)")
    parser.add_argument('--num-steps', type=int, dest='num_steps',
                        help="Time step", required=True)
    parser.add_argument('-bw', '--kernel-bandwidth', type=float, dest='bw', nargs='+',
                        help="Bandwidth for gaussian kernels")
    parser.add_argument('--birth-death-stride', type=int, dest='bd_stride',
                        help="Stride for the birth-death processes", default=0)
    parser.add_argument('--random-seed', type=int, dest='seed',
                        help="Seed for the random number generator")
    parser.add_argument('--trajectory-files', dest='traj_files',
                        help="Filename (prefix) to write the trajectories to. Will not save them if empty")
    parser.add_argument('--fes-file', dest='fes_file',
                        help="Save fes data to given path")
    parser.add_argument('--fes-image', dest='fes_image',
                        help="Save fes image to given path. If not specified it will show the image and ask for the name.")
    parser.add_argument('--tilt', type=float,
                        help="Tilt of the one-dimensional potential (linear coefficient)")
    args = parser.parse_args()
    if args.bd_stride != 0 and args.bw is None:
        raise ValueError("Error: Bandwidth for birth-death kernels not specified. (-bw argument)")
    return args




def main():
    args = parse_cliargs()

    # double well potential with optional tilt
    pot = Potential(np.array([0, float(args.tilt or 0.0), -4, 0, 1]))

    # set up LD and add particles
    ld = bpld(pot,
              args.time_step,
              args.friction,
              args.kt,
              args.seed,
              )
    # minumum slightly changes with tilt, calculate from coefficients
    extrema = np.polynomial.polynomial.polyroots(*ld.pot.der) # includes also maximum
    # add particles, round in favor of first state
    for _ in range(int(np.ceil(args.num_walkers * args.walkers_frac))):
        ld.add_particle([extrema[0]])
    for _ in range(int(args.num_walkers * (1 - args.walkers_frac))):
        ld.add_particle([extrema[2]])

    if args.seed is not None:
        args.seed += 1000
    bdld = BirthDeathLangevinDynamics(ld,
                                      args.bd_stride,
                                      args.bw,
                                      args.seed,
                                      )

    print(f'Running for {args.num_steps} timesteps with a birth/death stride of {args.bd_stride}')
    bdld.run(args.num_steps)

    print("\nFinished simulation")
    bdld.print_stats()

    # save trajectories to files
    if args.traj_files:
        print(f"Saving trajectories to: {args.traj_files}.i")
        bdld.save_trajectories(args.traj_files)

    # flatten trajectory for histogramming
    comb_traj = [pos for p in bdld.traj for pos in p]

    fes, axes = analysis.calculate_fes(comb_traj, args.kt,[(-2.5,2.5)], bins=201)
    ref = analysis.calculate_reference(ld.pot, np.array(axes).T)
    if args.fes_file:
        print(f"Saving fes to: {args.fes_file}")
        header = bdld.generate_fileheader(['pos', 'fes'])
        np.savetxt(args.fes_file, np.vstack((axes,fes)).T, fmt='%14.9f', header=str(header),
                   comments='', delimiter=' ', newline='\n')
    if args.fes_image:
        print(f"Saving fes image to: {args.fes_image}")
    plot_title = 'bw '+str(args.bw[0]) if args.bd_stride != 0 else None
    analysis.plot_fes(fes, axes, ref, fesrange=[-0.5,8.0], filename=args.fes_image, title=plot_title)

    # simply divide states by the 0 line for deltaF analysis
    delta_F_masks = [np.where(axes[0] < 0, True, False), np.where(axes[0] > 0, True, False)]
    delta_F = analysis.calculate_delta_F(fes, args.kt, delta_F_masks)[0]
    delta_F_ref = analysis.calculate_delta_F(ref, args.kt, delta_F_masks)[0]
    print(f'Delta F: {delta_F:.4} (ref: {delta_F_ref:.4})')


if __name__ == '__main__':
    main()