from site_forecast_app.save.utils import limit_adjuster


def test_limit_adjuster():

    # test no change
    assert limit_adjuster(0.08, 0.9, 2000) == 0.08

    # test limit is 10% of forecast
    assert limit_adjuster(0.1, 0.5, 1000) == 0.05

    # test limit is 1000 MW
    assert limit_adjuster(0.2, 0.5, 1e6) == 1000/1e6

    # test limit is 15% of forecast (non default values)
    assert limit_adjuster(0.2, 0.5, 2000, adjuster_limit_fraction=0.15) == 0.5*0.15

    # test limit is 2000 MW (non default values)
    assert limit_adjuster(0.2, 0.5, 1e6, adjuster_limit_mw=2000) == 2000/1e6
