"""Site Forecast App."""  # noqa: D104

from importlib.metadata import version, PackageNotFoundError


try:
    __version__ = version("site-forecast-app")
except PackageNotFoundError:
    __version__ = "v?"

