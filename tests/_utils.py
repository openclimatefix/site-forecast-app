"""Testing utils."""

import typer
from typer.testing import CliRunner


def run_typer_script(app_func, args: list[str], catch_exceptions: bool = False):
    """Util to test typer scripts while showing the stdout."""

    runner = CliRunner()

    # Create a temporary Typer app for testing
    if callable(app_func) and not hasattr(app_func, "_add_completion"):
        temp_app = typer.Typer()
        temp_app.command()(app_func)
        app_to_test = temp_app
    else:
        app_to_test = app_func
    # We catch the exception here no matter what, but we'll reraise later if need be.
    result = runner.invoke(app_to_test, args, catch_exceptions=True)
    # Without this the output to stdout/stderr is grabbed by click's test runner.
    # print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception and not catch_exceptions:
        raise result.exception

    return result
