"""Check the time difference between drawing a random particle from a list or just a random int"""

import timeit
import numpy as np


time_list = timeit.Timer(
    stmt="np.random.choice([i for i in range(num_part) if i not in excl])",
    setup="import numpy as np; num_part=50; excl=[np.random.randint(num_part)]",
).repeat(100, 10000)

print(
    f"Time w list (min, min, max): {min(time_list)}, {np.mean(time_list)}, {max(time_list)}",
    flush=True,
)

setup_func = """
import numpy as np; num_part=50; excl=np.random.randint(num_part)

def random_particle(num_part: int, excl: int) -> int:
    num = np.random.randint(num_part - 1)
    if num >= excl:
        num += 1
    return num
"""

time_int = timeit.Timer(
    stmt="random_particle(num_part, excl)",
    setup=setup_func,
).repeat(100, 10000)

print(
    f"Time w int (min, min, max): {min(time_int)}, {np.mean(time_int)}, {max(time_int)}",
    flush=True,
)
