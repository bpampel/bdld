"""Check the time 100 LD steps need for different numbers of particles"""
from contextlib import redirect_stdout
import io
import timeit

import numpy as np

setup = """
import numpy as np
from bdld.bussi_parinello_ld import BussiParinelloLD
from bdld.potential import Potential

# set up BussiParinelloLD with double well potential
pot = Potential(np.array([0, 0, -4, 0, 1]))
ld = BussiParinelloLD(pot, 0.005, 10, 0.66, 123654)

extrema = np.polynomial.polynomial.polyroots(*ld.pot.der)
for _ in range(n_particles // 2):
    ld.add_particle([extrema[0]])
for _ in range(n_particles // 2):
    ld.add_particle([extrema[2]])
"""

n_particles = [10, 100, 1000]

print("Times for 100 LD steps (min, min, max)")

for n in n_particles:
    with redirect_stdout(io.StringIO()):
        time = timeit.Timer(
            stmt="ld.step()", setup="n_particles={}\n".format(n) + setup
        ).repeat(10, 100)
    print(
        f"{n} particles: {min(time)}, {np.mean(time)}, {max(time)}",
        flush=True,
    )
