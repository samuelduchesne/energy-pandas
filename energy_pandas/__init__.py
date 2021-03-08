"""energy-pandas Module."""

from outdated import warn_if_outdated

from .energypandas import EnergySeries, EnergyDataFrame
from pkg_resources import get_distribution, DistributionNotFound

# Version of the package
try:
    __version__ = get_distribution("archetypal").version
except DistributionNotFound:
    # package is not installed
    __version__ = "0.0.0"  # should happen only if package is copied, not installed.

# warn if a newer version of archetypal is available
warn_if_outdated("energy-pandas", __version__)

__all__ = ["EnergySeries", "EnergyDataFrame"]
