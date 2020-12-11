"""Birth death algorithm"""

import copy
from typing import List, Optional, Union, Tuple

import numpy as np

from bdld.action import Action
from bdld.bussi_parinello_ld import BpldParticle
from bdld import grid


def dens_kernel_convolution(
    eq_density: grid.Grid, bw: np.ndarray, conv_mode: str
) -> grid.Grid:
    """Return convolution of the equilibrium probability density with the kernel

    If the "valid" conv_mode is used the returned grid is smaller than the original one.
    The "same" mode will return a grid with the same ranges, but might have issues
    due to edge effects from the convolution

    :param eq_density: grid with equilibrium probability density of system
    :param bw: bandwidths of the kernel (sigma)
    :param conv_mode: convolution mode to use (affects output size).
                      If 'valid' the resulting correction grid will have a
                      smaller domain than the original eq_density one.
                      If 'same' it will use exactly the ranges of the original grid.
    :return conv: grid holding the convolution values
    """
    kernel_ranges = [(-x, x) for x in 5 * bw]  # cutoff at 5 sigma
    kernel = grid.from_stepsizes(kernel_ranges, eq_density.stepsizes)
    kernel.data = kernel_sq_dist(kernel.points() ** 2, bw)
    # direct method is needed to avoid getting negative values instead of really small ones
    return grid.convolve(eq_density, kernel, mode=conv_mode, method="direct")


def calc_prob_correction_kernel(
    eq_density: grid.Grid, bw: np.ndarray, conv_mode: str = "same"
) -> grid.Grid:
    """Additive correction for the probabilites due to the Gaussian Kernel

    Calculates the following two terms from the Kernel K and equilibrium walker distribution pi
    .. :math::
        -log((K(x) * \pi(x)) / \pi(x)) + \int (log((K(x) * \pi(x)) / \pi(x))) \pi \mathrm{d}x

    :param eq_density: grid with equilibrium probability density of system
    :param bw: bandwidths of the kernel (sigma)
    :param conv_mode: convolution mode to use. See dens_kernel_convolution() for details.
    :return correction: grid wih the correction values
    """
    # setup kernel grid
    conv = dens_kernel_convolution(eq_density, bw, conv_mode)
    if conv_mode == "valid":
        # "valid" convolution shrinks grid --> shrink density as well
        dens_smaller = conv.copy_empty()
        dens_smaller.data = eq_density.interpolate(dens_smaller.points(), "linear")
        eq_density = dens_smaller
    log_term = np.log(conv / eq_density)
    integral_term = nd_trapz(log_term.data * eq_density.data, conv.stepsizes)
    correction = -log_term + integral_term
    if any(n > 101 for n in correction.n_points):
        correction = grid.sparsify(correction, [101] * correction.n_dim, "linear")
    return correction


def kernel_sq_dist(dist: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Return kernel values from the squared distances to center

    Currently directly returns Gaussian kernel
    other kernels could later be implemented via string argument

    :param dist: array of shape (n_dist, n_dim) with distances per dimensions
    :param bw: bandwidth per dimension
    """
    return (
        1
        / ((2 * np.pi) ** (len(bw) / 2) * np.prod(bw))
        * np.exp(-np.sum(dist / (2 * bw ** 2), axis=1))
    )


def nd_trapz(data: np.ndarray, dx: Union[List[float], float]) -> float:
    """Calculate a multidimensional integral via recursive usage of the trapezoidal rule

    Uses numpy's trapz for the 1d integrals

    :param data: values to integrate
    :param dx: distances between datapoints per dimension
    :return integral: integral value
    """
    if isinstance(dx, list):
        if dx:  # list not empty
            return nd_trapz(nd_trapz(data, dx=dx[-1]), dx[:-1])
        else:  # innermost iteration gives empty list
            return data
    else:
        return np.trapz(data, dx=dx)


def walker_density(pos: np.ndarray, bw: np.ndarray, kde: bool = False) -> np.ndarray:
    """Calculate the local density at each walker (average kernel value)

    The actual calculations are done by the different _walker_density functions
    depending on the number of walkers and the kde switch.

    :param numpy.ndarray pos: positions of particles
    :param float bw: bandwidth parameter of kernel
    :return numpy.ndarray kernel: kernel value matrix
    """
    if kde:
        return _walker_density_kde(pos, bw)
    elif len(pos) <= 10000:  # pdist matrix with maximum 10e8 float64 values
        return _walker_density_pdist(pos, bw)
    else:
        return _walker_density_manual(pos, bw)


def _walker_density_manual(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Calculate the local density at each walker manually for each walker

    This should be slower than the other variants but requires less memory
    because it is done on a per-walker basis

    :param pos: positions of particles
    :param bw: bandwidth parameter of kernel
    :return density: estimated density at each walker
    """
    n_part = len(pos)
    density = np.empty((n_part))
    for i in range(n_part):
        dist = np.array([(pos[i] - pos[j]) ** 2 for j in range(n_part) if j != i])
        kernel_values = kernel_sq_dist(dist, bw)
        density[i] = np.mean(kernel_values)
    return density


def _walker_density_kde(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Calculate the local density at each walker via KDE from statsmodels

    :param pos: positions of particles
    :param bw: bandwidth parameter of kernel
    :return density: estimated density at each walker
    """
    if pos.shape[1] == 1:
        from statsmodels.nonparametric.kde import KDEUnivariate

        kde = KDEUnivariate(pos)
        kde.fit(bw=bw)
        return kde.evaluate(pos.T)
    else:
        # if I understand the code correctly the multivariate form is just doing exactly what
        # I manually tested: just calculate all gaussian distances manually and _not_ in batches
        from statsmodels.nonparametric.kernel_density import KDEMultivariate

        var_type = "c" * pos.shape[1]  # continuous variables
        kde = KDEMultivariate(pos, var_type, bw=bw)
        return kde.pdf()


def _walker_density_pdist(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Calculate the local density at each walker via scipy's pdist

    Uses scipy to calculate a spare distance matrix between all walkers.
    Returns the sum over the Gaussian contributions at each walker.
    Note that the distance matrix becomes very large for many walkers,
    so this is only recommended for up to around 10,000 walkers.

    :param pos: positions of particles
    :param bw: bandwidth parameter of kernel
    :return density: estimated density at each walker
    """
    from scipy.spatial.distance import pdist, squareform

    n_dim = pos.shape[1]
    if n_dim == 1:  # faster version for 1d, otherwise identical
        dist = pdist(pos, "sqeuclidean")
        gauss = 1 / (np.sqrt(2 * np.pi) * bw[0]) * np.exp(-dist / (2 * bw[0] ** 2))
        return np.mean(squareform(gauss), axis=0)
    else:
        n_part = pos.shape[0]
        gauss_per_dim = np.empty((n_dim, (n_part * (n_part - 1)) // 2), dtype=np.double)
        for i in range(
            n_dim
        ):  # significantly faster variant than calling the kernel function
            dist = pdist(pos, "sqeuclidean")
            gauss_per_dim[i] = (
                1 / (np.sqrt(2 * np.pi) * bw[i]) * np.exp(-dist / (2 * bw[i] ** 2))
            )
        gauss = np.prod(gauss_per_dim, axis=0)
        return np.mean(squareform(gauss), axis=0)


class BirthDeath(Action):
    """Birth death algorithm

    :param correction: Grid holding the correction values
    :param kill_count: number of succesful death events
    :param kill_attempts: number of attempted death events
    :param dup_count: number of succesful birth events
    :param dup_attempts: number of succesful birth events
    """

    def __init__(
        self,
        particles: List[BpldParticle],
        dt: float,
        stride: int,
        bw: Union[List[float], np.ndarray],
        kt: float,
        correction_variant: Optional[str] = None,
        eq_density: Optional[grid.Grid] = None,
        seed: Optional[int] = None,
        stats_stride: Optional[int] = None,
        kde: Optional[bool] = None,
    ) -> None:
        """Set arguments

        :param particles: list of Particles shared with MD
        :param dt: timestep of MD
        :param stride: number of timesteps between birth-death exectutions
        :param bw: bandwidth for gaussian kernels per direction
        :param kt: thermal energy of system
        :param correction_variant: correction from original algorithm
                                   can be "additive", "multiplicative" or None
        :param eq_density: Equilibrium probability density of system, (grid, values)
        :param seed: Seed for rng (optional)
        :param stats_stride: Print statistics every n time steps
        :param kde: Use KDE from statsmodels to estimate walker density
        """
        self.particles: List[BpldParticle] = particles
        self.stride: int = stride
        self.dt: float = dt
        self.bw: np.ndarray = np.array(bw, dtype=float)
        self.inv_kt: float = 1 / kt
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.stats_stride: Optional[int] = stats_stride
        self.kde: bool = kde if kde else False
        print(
            f"Setting up birth/death scheme\n"
            f"Parameters:\n"
            f"  dt = {self.dt}\n"
            f"  bw = {self.bw}\n"
            f"  kt = {kt}"
        )
        if seed:
            print(f"  seed = {seed}")
        if kde:
            print(f"  using KDE to calculate kernels")
        print()
        self.correction_variant: Optional[str] = correction_variant
        if self.correction_variant:
            if not eq_density:
                raise ValueError("No equilibrium density for the correction was passed")
            if self.correction_variant == "additive":
                print("  using the additive correction")
                self.correction: grid.Grid = calc_prob_correction_kernel(
                    eq_density, self.bw, "same"
                )
            elif self.correction_variant == "multiplicative":
                print("  using the multiplicative correction")
                conv = dens_kernel_convolution(eq_density, self.bw, "same")
                self.correction = -np.log(
                    grid.sparsify(conv, [101] * conv.n_dim, "linear")
                )
            else:
                raise ValueError(
                    f"Specified correction variant {self.correction_variant} was not understood"
                )
        self.reset_stats()

    def run(self, step: int) -> None:
        """Perform birth-death step on particles

        :param step: current timestep of simulation
        """
        if step % self.stride == 0:
            bd_events = self.calculate_birth_death()
            for dup, kill in bd_events:
                self.particles[kill] = copy.deepcopy(self.particles[dup])
                # this copies all properties: is this desired?
                # what should be done with the momentum? Keep? Set to 0?
                # -> violates energy conservation!
                # keep the old random number for the initial thermostat step or generate new?
        if self.stats_stride and step % self.stats_stride == 0:
            self.print_stats()

    def calculate_birth_death(self) -> List[Tuple[int, int]]:
        """Calculate which particles to kill and duplicate

        The returned tuples are ordered, so the first particle in the tuple
        should be replaced by a copy of the second one

        :return bd_events: list of tuples with particles to duplicate and kill per event
        """
        num_part = len(self.particles)
        dup_list: List[int] = []
        kill_list: List[int] = []
        beta = self.calc_betas()
        # get number of attempts from betas
        curr_kill_attempts = np.count_nonzero(beta > 0)
        self.kill_attempts += curr_kill_attempts
        self.dup_attempts += num_part - curr_kill_attempts

        # evaluate all at same time not sequentially as in original paper
        # does it matter?
        prob = 1 - np.exp(-np.abs(beta) * self.dt)
        rand = self.rng.random(num_part)
        event_particles = np.where(rand <= prob)[0]
        self.rng.shuffle(event_particles)
        for i in event_particles:
            if i not in kill_list:
                if beta[i] > 0:
                    kill_list.append(i)
                    dup_list.append(self.random_particle(num_part, i))
                    self.kill_count += 1
                elif beta[i] < 0:
                    dup_list.append(i)
                    # prevent killing twice
                    kill_list.append(self.random_particle(num_part, i))
                    self.dup_count += 1

        return list(zip(dup_list, kill_list))

    def calc_betas(self) -> np.ndarray:
        """Calculate the birth/death rate for every particle"""

        pos = np.array([p.pos for p in self.particles])
        with np.errstate(divide="ignore"):
            # density can be zero and make beta -inf. Filter when averaging in next step
            beta = np.log(walker_density(pos, self.bw, self.kde))

        if self.correction_variant == "additive" or not self.correction_variant:
            ene = np.array([p.energy for p in self.particles])
            beta += ene * self.inv_kt
            beta -= np.mean(beta[beta != -np.inf])
            if self.correction_variant == "additive":
                beta += self.correction.interpolate(pos, "linear").reshape(len(pos))
        elif self.correction_variant == "multiplicative":
            # do not use actual energies, just add the smoothed density
            beta += self.correction.interpolate(pos, "linear").reshape(len(pos))
            beta -= np.mean(beta[beta != -np.inf])
        return beta

    def random_particle(self, num_part: int, excl: int) -> int:
        """Select random particle while excluding current one

        :param num_part: total number of particles
        :param excl: particle to exclude
        :return num: random particle
        """
        num = self.rng.integers(num_part - 1)
        if num >= excl:
            num += 1
        return num

    def walker_density_grid(self, grid: np.ndarray, energy: np.ndarray) -> np.ndarray:
        """Calculate the density of walkers and bd-probabilities on a grid

        :param grid: positions to calculate the kernel values
        :param grid: energies of the grid values
        :return array: grid rho beta
        """
        rho = []
        beta = []
        walker_pos = [p.pos for p in self.particles]
        walker_ene = [p.energy for p in self.particles]
        for g, e in zip(grid, energy):
            pos = np.append(g, walker_pos).reshape(
                (len(walker_pos) + 1, walker_pos[1].ndim)
            )  # array with positions as subarrays
            ene = np.append(e, walker_ene)
            # full kernel is needed for probability (normalization)
            rho_g = walker_density(pos, self.bw)
            rho.append(rho_g[0])

            beta_g = np.log(rho_g) + ene * self.inv_kt
            beta.append(beta_g[0] - np.mean(beta_g[beta_g != -np.inf]))

        return np.c_[grid, rho, beta]

    def print_stats(self, reset: bool = False) -> None:
        """Print birth/death probabilities to screen"""
        try:
            kill_perc = 100 * self.kill_count / self.kill_attempts
        except ZeroDivisionError:
            kill_perc = np.nan
        try:
            dup_perc = 100 * self.dup_count / self.dup_attempts
        except ZeroDivisionError:
            dup_perc = np.nan
        try:
            ratio_succ = self.dup_count / self.kill_count
        except ZeroDivisionError:
            ratio_succ = np.nan
        try:
            ratio_attempts = self.dup_attempts / self.kill_attempts
        except ZeroDivisionError:
            ratio_attempts = np.nan
        print(
            f"Succesful birth events: {self.dup_count}/{self.dup_attempts} ({dup_perc:.4}%)"
        )
        print(
            f"Succesful death events: {self.kill_count}/{self.kill_attempts} ({kill_perc:.4}%)"
        )
        print(
            f"Ratio birth/death: {ratio_succ:.4} (succesful)  {ratio_attempts:.4} (attemps)"
        )
        if reset:
            self.reset_stats()

    def reset_stats(self) -> None:
        """Set all logging counters to zero"""
        self.dup_count = 0
        self.dup_attempts = 0
        self.kill_count = 0
        self.kill_attempts = 0
