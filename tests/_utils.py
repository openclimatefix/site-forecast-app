"""Testing utils."""

import typer
from typer.testing import CliRunner


def run_typer_script(app_func, args: list[str], catch_exceptions: bool = False):
    """Util to test typer scripts while showing the stdout."""

    runner = CliRunner()

    # Create a temporary Typer app for testing
    if callable(app_func) and not hasattr(app_func, '_add_completion'):
        temp_app = typer.Typer()
        temp_app.command()(app_func)
        app_to_test = temp_app
    else:
        app_to_test = app_func

    result = runner.invoke(app_to_test, args, catch_exceptions=True)
    if result.exception and not catch_exceptions:
        raise result.exception

    return result
