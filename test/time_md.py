"""Check the time difference between calculating the full kernel distance matices or each row independently"""

import timeit
import numpy as np

setup = '''
import numpy as np
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.potential import Potential

# parameters to change
n_particles = 1000

# set up BussiParinelloLD with double well potential
pot = Potential(np.array([0, 0, -4, 0, 1]))
ld = BussiParinelloLD(pot, 0.005, 10, 0.66, 123654)

extrema = np.polynomial.polynomial.polyroots(*ld.pot.der)
for _ in range(n_particles // 2):
    ld.add_particle([extrema[0]])
for _ in range(n_particles // 2):
    ld.add_particle([extrema[2]])
'''

time = timeit.Timer(stmt="ld.step()", setup=setup).repeat(10, 100)
print(
    f"Time (min, min, max): {min(time)}, {np.mean(time)}, {max(time)}",
    flush=True,
)
