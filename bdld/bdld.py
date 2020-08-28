"""Main bdld algorithm file hosting the BirthDeathLangevinDynamics class"""

from typing import List, Optional, Tuple, Union
import numpy as np

from bdld.helpers.plumed_header import PlumedHeader as PlmdHeader

from bdld import analysis
from bdld.birth_death import BirthDeath
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.histogram import Histogram


class BirthDeathLangevinDynamics:
    """Combine Langevin dynamics with birth/death events

    :param ld: completely set up BussiParinelloLD instance
    :param bd: BirthDeath instance that will be generated by the setup() function
    :param bd_stride: number of ld steps between execution of the birth/death algorithm, defaults to 0 (no bd)
    :param bd_time_step: time between bd events in units of the ld
    :param bd_seed: seed for the RNG of the birth/death algorithm
    :param bd_bw: bandwidth for gaussian kernels per direction used in birth/death
    :param traj: trajectories of all particles
    :param steps_since_bd: counts the passed steps since the last execution of the birth/death algorithm
    """

    def __init__(
        self,
        ld: BussiParinelloLD,
        bd_stride: int = 0,
        bd_bw: List[float] = [0.0],
        bd_seed: Optional[int] = None,
        kde: bool = False,
    ) -> None:
        """Generate needed varables and the birth/death instance from arguments"""
        self.ld: BussiParinelloLD = ld
        self.bd: Optional[BirthDeath] = None
        self.bd_stride: int = bd_stride
        self.bd_time_step: float = ld.dt * bd_stride
        self.bd_seed: Optional[int] = bd_seed
        self.bd_bw: List[float] = bd_bw
        self.traj: List[List[np.ndarray]] = []
        self.steps_since_bd: int = 0
        self.histo: Optional[Histogram] = None
        self.histo_stride: int = 0
        self.kde: bool = kde
        self.setup()

    def setup(self) -> None:
        """Set up bd and initialize trajectory lists"""
        if self.bd_stride != 0:
            if any(bw <= 0 for bw in self.bd_bw):
                raise ValueError(
                    f"The bandwidth of the Gaussian kernels needs"
                    f"to be greater than 0 (is {self.bd_bw})"
                )
            self.setup_bd()
        # initialize trajectory list
        self.traj = [[np.copy(p.pos)] for p in self.ld.particles]

    def setup_bd(self) -> None:
        """Set up birth death from parameters"""
        if self.bd_stride != 0:
            self.bd = BirthDeath(
                self.ld.particles,
                self.bd_time_step,
                self.bd_bw,
                self.ld.kt,
                self.bd_seed,
                True,
                self.kde,
            )

    def init_histogram(
        self, n_bins: List[int], ranges: List[Tuple[float, float]], stride=None
    ) -> None:
        """Initialize a histogram for the trajectories

        :param n_bins: number of bins of the histogram per dimension
        :param ranges: extent of histogram (min, max) per dimension
        :param int stride: add trajectory to the histogram every n steps
        """
        if self.ld.pot.n_dim != len(n_bins):
            e = (
                "Dimensions of histogram bins don't match dimensions of system "
                + f"({len(n_bins)} vs. {self.ld.pot.n_dim})"
            )
            raise ValueError(e)
        if self.ld.pot.n_dim != len(ranges):
            e = (
                "Dimensions of histogram ranges don't match dimensions of system "
                + f"({len(ranges)} vs. {self.ld.pot.n_dim})"
            )
            raise ValueError(e)
        self.histo = Histogram(n_bins, ranges)
        self.histo_stride = stride
        if self.histo_stride is None:
            # add to histogram every 1,000,000 trajectory points by default
            self.histo_stride = 1000000 / len(self.ld.particles)

    def add_trajectory_to_histogram(self, clear_traj: bool) -> None:
        """Add trajectory data to histogram

        :param bool clear_traj: delete the trajectory data after adding to histogram
        """
        if not self.histo:
            raise ValueError("Histogram was not initialized yet")
        comb_traj = np.vstack([pos for part in self.traj for pos in part])
        self.histo.add(comb_traj)
        if clear_traj:
            self.traj = [[] for i in range(len(self.ld.particles))]

    def run(self, num_steps: int) -> None:
        """Run the simulation for given number of steps

        It performs first the Langevin dynamics, then optionally the birth/death
        steps and then saves the positions of the particles to the trajectories.

        The num_steps argument takes the previous steps into account for the
        birth/death stride. For example having a bd_stride of 10 and first
        running for 6 and then 7 steps will perform a birth/death step on the
        4th step of the second run.

        :param int num_steps: Number of steps to run
        """
        for i in range(self.steps_since_bd + 1, self.steps_since_bd + 1 + num_steps):
            self.ld.step()
            if self.bd:
                self.bd.step()
            for j, p in enumerate(self.ld.particles):
                self.traj[j].append(np.copy(p.pos))
            if self.histo and i % self.histo_stride == 0:
                self.add_trajectory_to_histogram(True)
        # increase counter only once
        if self.bd:
            self.steps_since_bd = (self.steps_since_bd + num_steps) % self.bd_stride

    def save_analysis_grid(
        self, filename: str, grid: Union[List[np.ndarray], np.ndarray]
    ) -> None:
        """Analyse the values of rho and beta on a grid

        :param filename: path to save grid to
        :param grid: list or numpy array with positions to calculate values
        """
        if not self.bd:
            raise ValueError("No birth/death to analize")
        ana_ene = [self.ld.pot.evaluate(p)[0] for p in grid]
        ana_values = self.bd.prob_density_grid(grid, ana_ene)
        header = self.generate_fileheader(["pos", "rho", "beta"])
        np.savetxt(
            filename,
            ana_values,
            fmt="%14.9f",
            header=str(header),
            comments="",
            delimiter=" ",
            newline="\n",
        )

    def save_trajectories(self, filename: str) -> None:
        """Save all trajectories to files

        :param filename: basename for files, is appended by '.i' for the individual files
        """
        for i, t in enumerate(self.traj):
            header = self.generate_fileheader([f"traj.{i}"])
            np.savetxt(
                filename + "." + str(i),
                t,
                header=str(header),
                comments="",
                delimiter=" ",
                newline="\n",
            )

    def save_fes(self, filename: str) -> None:
        """Calculate FES and save to text file

        This does histogramming of the trajectories first if necessary

        :param string filename: path to save FES to
        """
        if not self.histo:
            raise ValueError("Histogram for FES needs to be initialized first")
        if any(t for t in self.traj):
            self.add_trajectory_to_histogram(True)
        fes, pos = self.histo.calculate_fes(self.ld.kt)
        header = self.generate_fileheader(["pos fes"])
        data = np.vstack((pos, fes)).T
        np.savetxt(
            filename, data, header=str(header), comments="", delimiter=" ", newline="\n"
        )

    def plot_fes(
        self,
        filename: Optional[str] = None,
        plot_domain: Optional[Tuple[float, float]] = None,
        plot_title: Optional[str] = None,
    ) -> None:
        """Plot fes with reference and optionally save to file

        :param filename: optional filename to save figure to
        :param plot_domain: optional list with minimum and maximum value to show
        :param plot_title: optional title for the legend
        """
        if not self.histo:
            raise ValueError("Histogram for FES needs to be initialized first")
        if any(t for t in self.traj):
            self.add_trajectory_to_histogram(True)
        if self.histo.fes is None:
            self.histo.calculate_fes(self.ld.kt)
        analysis.plot_fes(
            self.histo.fes,
            self.histo.bin_centers(),
            # temporary fix, needs to be changed for more than 1d
            ref=self.ld.pot.calculate_reference(self.histo.bin_centers()[0]),
            plot_domain=plot_domain,
            filename=filename,
            title=plot_title,
        )

    def generate_fileheader(self, fields: List[str]) -> PlmdHeader:
        """Get plumed-style header from variables to print with data to file

        :param fields: list of strings for the field names (first line of header)
        :return header:
        """
        header = PlmdHeader(
            [
                " ".join(["FIELDS"] + fields),
                f"SET dt {self.ld.dt}",
                f"SET kt {self.ld.kt}",
                f"SET friction {self.ld.friction}",
                f"SET bd_stride {self.bd_stride}",
                f"SET bd_bandwidth {self.bd_bw}",
            ]
        )
        return header
