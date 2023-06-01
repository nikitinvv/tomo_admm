from pkg_resources import get_distribution, DistributionNotFound

from tomo_admm.radonusfft import *
from tomo_admm.solver_tomo import *
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass