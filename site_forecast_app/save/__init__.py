"""Public API for the save subpackage.

Re-exports everything external callers (app.py, tests) already import
from ``site_forecast_app.save``, so no change is needed at call sites.
"""

from grpclib.client import Channel  # noqa: F401  (used by integration test monkeypatching)

from dp_sdk.ocf import dp  # noqa: F401  (used by integration test monkeypatching)

from site_forecast_app.save.data_platform import (
    DataPlatformClient,
    build_dp_location_map,
    fetch_dp_location_map,
    get_dataplatform_client,
    make_forecaster_adjuster,
    save_forecast_to_dataplatform,
    save_to_dataplatform,
)
from site_forecast_app.save.save import save_forecast
from site_forecast_app.save.utils import limit_adjuster  # noqa: F401

__all__ = [
    "Channel",
    "DataPlatformClient",
    "build_dp_location_map",
    "dp",
    "fetch_dp_location_map",
    "get_dataplatform_client",
    "limit_adjuster",
    "make_forecaster_adjuster",
    "save_forecast",
    "save_forecast_to_dataplatform",
    "save_to_dataplatform",
]
