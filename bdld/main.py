"""Class that takes care about running and analysing a simulation"""

import argparse
from typing import Dict, List, Optional
import sys

import numpy as np

from bdld import inputparser
from bdld.action import Action
from bdld.potential import Potential
from bdld.birth_death import BirthDeath
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.grid import Grid
from bdld.optional_actions import TrajectoryAction, HistogramAction, FesAction
from bdld.helpers.plumed_header import PlumedHeader as PlmdHeader
from bdld.helpers.misc import backup_if_exists


def main() -> None:
    """Main function of simulation

    Does these things (in order):
    1) parse cli args
    2) get input from file
    3) set up all actions
    4) run the main loop for the desired time steps
    5) run final actions
    """
    # parse cli argument(s)
    cliargs = argparse.ArgumentParser()
    cliargs.add_argument("input", type=str)
    infile = cliargs.parse_args().input

    try:
        config = inputparser.Input(infile)
    except FileNotFoundError:
        print(f"Error: Input file '{infile}' could not be found", file=sys.stderr)
        sys.exit(1)
    except inputparser.InputError as e:
        print(e.args[0], file=sys.stderr)
        sys.exit(1)


    # initialize all actions
    actions: Dict[str, Action] = {}
    # dicts are ordered since python 3.7
    # actions are performed in order of insertion to the dict
    # make sure actions are added after their dependencies (e.g. fes after hist)

    pot = setup_pot(config.potential)
    ld = BussiParinelloLD(
        pot,
        config.ld["timestep"],
        config.ld["friction"],
        config.ld["kt"],
        config.ld["seed"],
    )
    init_particles(config.particles, ld)
    actions["ld"] = ld

    if config.birth_death:
        actions["birth_death"] = setup_birth_death(config.birth_death, ld)

    # main loop
    for step in range(config.ld["n_steps"]):
        for action in actions.values():
            action.run(step)

    # final_run
    for action in actions.values():
        action.final_run(config.ld["n_steps"] - 1)


def setup_pot(options: Dict) -> Potential:
    """Return potential from given options"""
    if options["n_dim"] == 1:
        ranges = [options["min"], options["max"]]
    else:
        ranges = list(zip(options["min"], options["max"]))
    return Potential(options["coeffs"], ranges)


def init_particles(options: Dict, ld: BussiParinelloLD) -> None:
    """Add particles to ld with the given algorithm

    random-global: distribute randomly on full range of potential
    random-pos: distribute randomly on the given positions
    fractions-pos: distribute with given fractions on the given positions
    """
    if options["initial-distribution"] == "random-global":
        rng = np.random.default_rng(options["seed"])
        for _ in range(options["number"]):
            pos = [rng.uniform(start, end) for start, end in ld.pot.ranges]
            ld.add_particle(pos)
    elif options["initial-distribution"] == "random-pos":
        rng = np.random.default_rng(options["seed"])
        init_pos_choices = [pos for key, pos in options.items() if "pos" in key]
        for _ in range(options["number"]):
            ld.add_particle(rng.choice(init_pos_choices, axis=0))
    elif options["initial-distribution"] == "fractions-pos":
        counts = [int(frac * options["number"]) for frac in options["fractions"]]
        counts[0] += options["number"] - np.sum(counts)  # rounding offset
        init_pos_choices = [pos for key, pos in options.items() if "pos" in key]
        if len(init_pos_choices) != len(counts):
            e = "fractions in [particles]: number of positions and fractions do not match"
            raise ValueError(e)
        for i, pos in enumerate(init_pos_choices):
            for _ in range(counts[i]):
                ld.add_particle(pos)


def setup_birth_death(options: Dict, ld: BussiParinelloLD) -> BirthDeath:
    """Setup BirthDeath instance from options

    Currently this requires to get the true probability density values from the
    potential for the corrections of the algorithm, so this is also set up here
    """
    bd_bw = np.array(options["kernel-bandwidth"])
    if options["correction_variant"]:
        eq_density: Optional[Grid] = bd_prob_density(ld.pot, bd_bw, ld.kt)
    else:
        eq_density = None
    return BirthDeath(
        ld.particles,
        ld.dt,
        options["stride"],
        bd_bw,
        ld.kt,
        options["correction_variant"],
        eq_density,
        options["seed"],
        options["stats-stride"],
    )


def bd_prob_density(pot: Potential, bd_bw: List[float], kt: float) -> Grid:
    """Return probability density grid needed for BirthDeath

    This is usually a unknown quantity, so this has to be replaced by an estimate
    in the future. E.g. enforce usage of the histogram and use that as estimate at
    current time with iterative updates

    Because of the current free choice of points, we use rather a lot and make
    sure the Kernel grid will have at least 20 points per dimension within 5 sigma

    :return prob_grid: Grid of the probability density
    """
    n_grid_points = []
    for dim, r in enumerate(pot.ranges):
        # check minimal number of points for 20 points within 5 sigma
        min_points_gaussian = int(np.ceil((r[1] - r[0]) / (0.5 * bd_bw[dim])))
        # the large number of points is only used to calculate the correction once
        tmp_grid_points = max(501, min_points_gaussian)
        if tmp_grid_points % 2 == 0:
            tmp_grid_points += 1  # odd number is better for convolution
        n_grid_points.append(tmp_grid_points)
    return pot.calculate_probability_density(kt, pot.ranges, n_grid_points)
