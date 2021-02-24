"""energy-pandas Module."""

from outdated import warn_if_outdated

from .energypandas import EnergySeries, EnergyDataFrame

# Version of the package
__version__ = "0.1.0"

# warn if a newer version of archetypal is available
warn_if_outdated("energy-pandas", __version__)

__all__ = ["EnergySeries", "EnergyDataFrame"]
