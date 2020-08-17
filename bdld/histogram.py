#!/usr/bin/env python3

import numpy as np


class Histogram():
    """Histogram data and calculate FES from the histogram

    Enables histogramming over time, i.e. adding more data to the existing histogram
    """

    def __init__(self, n_bins, ranges):
        """Set up empty histogram instance

        :param int n_bins: number of bins for histogramming
        :param ranges: extent of histogram (min, max) per dimension
        :type ranges: list of tuples
        :param bins: bin edges of the histogram
        :type bins: list of numpy.ndarray per dimension
        :param histo: the histogram data (counts) corresponding to the bins
        :type histo: list of numpy.ndarray per dimension
        :param fes: the free energy values corresponding to the histogram
        :type fes: list of numpy.ndarray per dimension
        """
        self.n_bins = n_bins
        self.ranges = ranges
        self.bins = []
        self.histo = []
        self.fes = None
        # create bins from arbitrary value, there doesn't seem to be a function doing it
        _, self.bins = np.histogramdd([0], bins=self.n_bins, range=self.ranges)

    def add(self, data):
        """Add data to histogram

        :param data: The values to add. If multidimensional should be either a list of lists or
        :type data: list (1d), list of lists or numpy.ndarrays (arbitrary dimensions)
        """
        tmp_histo, _ = np.histogramdd(np.vstack(data), bins=self.bins)
        self.histo += tmp_histo

    def bin_centers(self):
        """Calculate the centers of the histogram bins from the bin edges

        :return centered_bins: the centers of the histogram bins
        :type centered_bins: list with numpy.ndarray per dimension
        """
        return [np.array([(bins_x[i] + bins_x[i+1]) / 2 for i in range(0,len(bins_x)-1)]) for bins_x in self.bins]

    def calculate_fes(self, kt, mintozero=True):
        """Calculate free energy surface from histogram

        Overwrites the fes attribute from the class instance and returns the data
        in plottable form

        :param float kt: thermal energy of the system
        :param bool mintozero: shift FES to have minimum at zero

        :return fes: numpy array with free energy values
        :return pos: list of numpy arrays with the positions corresponding to the FES values
        """
        fes = np.where(self.histo == 0, np.inf, - kt * np.log(self.histo, where=(self.histo!=0)))
        if mintozero:
            fes -= np.min(fes)
        self.fes = fes
        return fes, self.bin_centers()
