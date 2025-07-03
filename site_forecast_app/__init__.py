"""Site Forecast App."""
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # For Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pvnet-app")
except PackageNotFoundError:
    __version__ = "v?"

