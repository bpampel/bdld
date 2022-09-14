"""Import all submodules"""
# when doing `import bdld.potential`
from . import mueller_brown
from . import polynomial
from . import potential

# when doing `from bdld.potential import *` (not recommended)
__all__ = ["mueller_brown", "polynomial", "potential"]
