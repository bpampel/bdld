"""Class that takes care about running and analysing a simulation"""

import argparse
from typing import Dict, List, Optional
import sys

import numpy as np

from bdld import actions, inputparser, potential
from bdld.grid import Grid

# alias shortcuts
Action = actions.action.Action
BirthDeath = actions.birth_death.BirthDeath
BussiParinelloLD = actions.bussi_parinello_ld.BussiParinelloLD
TrajectoryAction = actions.trajectory_action.TrajectoryAction
HistogramAction = actions.histogram_action.HistogramAction
FesAction = actions.fes_action.FesAction
DeltaFAction = actions.delta_f_action.DeltaFAction

Potential = potential.potential.Potential


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
    except inputparser.InputError as e:  # print error message and exit
        print(e.args[0], file=sys.stderr)
        sys.exit(1)

    # initialize all actions
    all_actions: Dict[str, Action] = {}
    # dicts are ordered since python 3.7
    # actions are performed in order of insertion to the dict
    # make sure actions are added after their dependencies (e.g. fes after hist)

    # compulsory: set up Langevin dynamics
    pot = setup_potential(config.potential)
    ld = setup_ld(config.ld, pot)
    init_particles(config.particles, ld)
    all_actions["ld"] = ld

    # optional actions in reasonable order
    try:
        if config.birth_death:
            all_actions["birth_death"] = setup_birth_death(config.birth_death, ld)
        if config.trajectories:
            all_actions["trajectories"] = setup_trajectories(config.trajectories, ld)
        if config.histogram:
            all_actions["histogram"] = setup_histogram(
                config.histogram, all_actions["trajectories"]
            )
        if config.fes:
            all_actions["fes"] = setup_fes(config.fes, all_actions["histogram"])
        if config.delta_f:
            all_actions["delta_f"] = setup_delta_f(config.delta_f, all_actions["fes"])
    except KeyError as e:
        print(
            f"Error: An action was specified that requires the '{e.args[0]}' section"
            " in input but it was not found",
            file=sys.stderr,
        )
        sys.exit(1)
    except inputparser.InputError as e:
        print(e.args[0], file=sys.stderr)
        sys.exit(1)

    n_steps = config.ld["n_steps"]
    print(f"Setup finished, now running for {n_steps} steps")
    # main loop
    for step in range(1, n_steps + 1):
        for action in all_actions.values():
            action.run(step)

    print("Simulation finished, performing final actions")
    for action in all_actions.values():
        action.final_run(n_steps)

    print("Finished without errors")


def setup_potential(options: Dict) -> Potential:
    """Return potential from given options"""
    if options["type"] == "polynomial":
        if options["n_dim"] == 1:
            ranges = [(options["min"], options["max"])]
            return potential.polynomial.PolynomialPotential(options["coeffs"], ranges)
        else:
            ranges = list(zip(options["min"], options["max"]))
            coeffs = potential.polynomial.coefficients_from_file(
                options["coeffs-file"], options["n_dim"]
            )
        return potential.polynomial.PolynomialPotential(coeffs, ranges)
    elif options["type"] == "mueller-brown":
        return potential.mueller_brown.MuellerBrownPotential(options["scaling-factor"])
    else:
        raise inputparser.InputError(
            f'Specified potential type "{options["type"]}" is not implemented',
            "type",
            "potential",
        )


def setup_ld(options: Dict, pot: Potential) -> BussiParinelloLD:
    """Return Langevin Dynamics with given options on the potential"""
    return BussiParinelloLD(
        pot,
        options["timestep"],
        options["friction"],
        options["kt"],
        options["seed"],
    )


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
        # normalize so sum of fractions is one -> allows also total number inputs
        normalized_fractions = np.array(options["fractions"]) / np.sum(
            options["fractions"]
        )
        counts = [int(frac * options["number"]) for frac in normalized_fractions]
        counts[0] += options["number"] - np.sum(counts)  # rounding offset
        # dicts are ordered since python 3.7: no need to actually parse pos1 etc
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
    if ld.pot.n_dim == 1:
        bd_bw = np.array([options["kernel-bandwidth"]])
    else:
        bd_bw = np.array(options["kernel-bandwidth"])
    if len(bd_bw) != ld.pot.n_dim:
        raise inputparser.InputError(
            f"dimensions of kernel bandwidth does not match potential (should be {ld.pot.n_dim} values)",
            "kernel-bandwidth",
            "birth-death",
        )

    if options["correction-variant"]:
        eq_density: Optional[Grid] = bd_prob_density(ld.pot, bd_bw, ld.kt)
    else:
        eq_density = None
    return BirthDeath(
        ld.particles,
        ld.dt,
        options["stride"],
        bd_bw,
        ld.kt,
        options["correction-variant"],
        eq_density,
        options["seed"] + 1000 if options["seed"] else None,
        options["stats-stride"],
        options["stats-filename"],
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


def setup_trajectories(options: Dict, ld: BussiParinelloLD) -> TrajectoryAction:
    """Setup TrajectoryAction on ld with given options"""
    return TrajectoryAction(
        ld,
        options["stride"],
        options["filename"],
        options["write-stride"],
        options["fmt"],
    )


def setup_histogram(options: Dict, traj_action: TrajectoryAction) -> HistogramAction:
    """Setup HistogramAction on TrajectoryAction with given options"""
    if traj_action.ld.pot.n_dim == 1:
        ranges = [(options["min"], options["max"])]
        n_bins = [options["bins"]]
    else:
        ranges = list(zip(options["min"], options["max"]))
        n_bins = options["bins"]
    return HistogramAction(
        traj_action,
        n_bins,
        ranges,
        options["stride"],
        options["filename"],
        options["write-stride"],
        options["fmt"],
    )


def setup_fes(options: Dict, histo_action: HistogramAction) -> FesAction:
    """Setup FesAction on HistogramAction with given options"""
    # don't assume this will stay here in the future, it's ugly
    ref = histo_action.traj_action.ld.pot.calculate_reference(
        histo_action.histo.points()
    )
    return FesAction(
        histo_action,
        options["stride"],
        options["filename"],
        options["write-stride"],
        options["fmt"],
        options["plot-stride"],
        options["plot-filename"],
        options["plot-domain"],
        options["plot-title"],
        ref,
    )


def setup_delta_f(options: Dict, fes_action: FesAction) -> DeltaFAction:
    """Setup FesAction on HistogramAction with given options"""
    min_list = inputparser.get_all_numbered_values(options, "state", "-min")
    max_list = inputparser.get_all_numbered_values(options, "state", "-max")
    state_ranges = inputparser.min_max_to_ranges(min_list, max_list)
    if len(state_ranges) < 2:
        e_msg = "You need to specify at least 2 states for delta F calculations"
        raise ValueError(e_msg)

    fes_points = fes_action.histo_action.histo.points()
    n_dim = fes_points.shape[-1]

    masks: List[np.ndarray] = []
    for state in state_ranges:
        # start with boolean array for first (0) dimension
        mask = (fes_points[:, 0] >= state[0][0]) & (fes_points[:, 0] <= state[1][0])
        try:
            for i in range(1, n_dim):  # now 'bitwise and' with all other dimensions
                mask = (
                    mask
                    & (fes_points[:, i] >= state[0][i])
                    & (fes_points[:, i] <= state[1][i])
                )
        except IndexError as e:
            raise inputparser.InputError(
                "The specified state dimensions are smaller than the fes dimensions",
                "state{i}-min",
                "delta-f",
            ) from e
        masks.append(mask)

    return DeltaFAction(
        fes_action,
        masks,
        options["stride"],
        options["filename"],
        options["write-stride"],
        options["fmt"],
    )
