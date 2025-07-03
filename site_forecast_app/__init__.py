"""Site Forecast App."""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("pvnet-app")
except PackageNotFoundError:
    __version__ = "v?"

