"""Check the time difference between calculating the full kernel distance matices or each row independently"""

import timeit
import numpy as np
from scipy.spatial.distance import pdist, sqeuclidean, squareform

setup = '''
import numpy as np
from bdld.actions import birth_death as bd

n_particles = 50
n_dim = 1
bw = np.ones(n_dim)*0.1

pos = np.random.random_sample((n_particles, n_dim))
'''

time_pdist = timeit.Timer(stmt="bd._walker_density_pdist(pos, bw)", setup=setup).repeat(
    100, 100
)
print(
    f"Time for pdist (min, min, max): {min(time_pdist)}, {np.mean(time_pdist)}, {max(time_pdist)}",
    flush=True,
)

time_kde = timeit.Timer(stmt="bd._walker_density_kde(pos, bw)", setup=setup).repeat(
    100, 100
)
print(
    f"Time for kde (min, min, max): {min(time_kde)}, {np.mean(time_kde)}, {max(time_kde)}"
)

time_manual = timeit.Timer(stmt="bd._walker_density_manual(pos, bw)", setup=setup).repeat(
    100, 100
)
print(
    f"Time for manual (min, min, max): {min(time_manual)}, {np.mean(time_manual)}, {max(time_manual)}"
)
