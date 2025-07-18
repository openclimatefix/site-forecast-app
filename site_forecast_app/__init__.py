"""Site Forecast App."""

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("site-forecast-app")
except PackageNotFoundError:
    __version__ = "v?"

