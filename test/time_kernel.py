"""Check the time difference between calculating the full kernel distance matices or each row independently"""

import timeit
import numpy as np
from scipy.spatial.distance import pdist, sqeuclidean, squareform

setup = '''
import numpy as np
from scipy.spatial.distance import pdist, sqeuclidean, squareform

def walker_density_pdist(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Variant relying on numpys pdist that creates the full matrix"""
    dist = pdist(pos, "sqeuclidean")
    gauss = (
        1 / (2 * np.pi * bw ** 2) ** (pos.ndim / 2) * np.exp(-dist / (2 * bw) ** 2)
    )
    return np.mean(squareform(gauss), axis=0)


def walker_density_rowwise(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Variant calculating the distances manually and row-wise"""
    density = np.empty((len(pos)))
    gauss_norm = 1 / (2 * np.pi * bw ** 2) ** (pos.ndim / 2)
    gauss_sigma = 1 / (2 * bw) ** 2
    for i in range(len(pos)):
        dist = np.fromiter(
            (sqeuclidean(pos[i], pos[j]) for j in range(len(pos)) if j != i),
            np.float64,
            len(pos) - 1,
        )
        gauss_dist = gauss_norm * np.exp(-dist * gauss_sigma)
        density[i] = np.mean(gauss_dist)
    return density


def walker_density_manual(pos: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Variant calculating the distances manually and row-wise"""
    n_part = len(pos)
    n_dim = len(pos[0])
    density = np.empty((len(pos)))
    gauss_norm = 1 / ((2 * np.pi) ** (pos.ndim / 2) * np.sqrt(np.prod(bw ** 2)))
    variance_fac = 1 / (2 * bw ** 2)
    for i in range(len(pos)):
        gauss_arg = [- np.sum([variance_fac[k] * (pos[i][k] - pos[j][k])**2 for k in range(n_dim)]) for j in range(n_part) if j != i]
        gauss_dist = gauss_norm * np.exp(gauss_arg)
        density[i] = np.mean(gauss_dist)
    return density


n_particles = 50
n_dim = 1
bw = np.array([0.1])

pos = np.random.random_sample((n_particles, n_dim))
'''

time_pdist = timeit.Timer(stmt="walker_density_pdist(pos, bw)", setup=setup).repeat(10, 100)
print(f"Time for pdist (min, min, max): {min(time_pdist)}, {np.mean(time_pdist)}, {max(time_pdist)}", flush=True)

time_rowwise = timeit.Timer(stmt="walker_density_rowwise(pos, bw)", setup=setup).repeat(10, 100)
print(f"Time for row-wise (min, min, max): {min(time_rowwise)}, {np.mean(time_rowwise)}, {max(time_rowwise)}")

time_manual = timeit.Timer(stmt="walker_density_manual(pos, bw)", setup=setup).repeat(10, 100)
print(f"Time for manual (min, min, max): {min(time_manual)}, {np.mean(time_manual)}, {max(time_manual)}")
