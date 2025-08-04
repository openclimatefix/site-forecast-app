"""Testing utils."""

from typer.testing import CliRunner


def run_typer_script(app, args: list[str], catch_exceptions: bool = False):
    """Util to test typer scripts while showing the stdout."""

    runner = CliRunner()

    # We catch the exception here no matter what, but we'll reraise later if need be.
    result = runner.invoke(app, args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by typer's test runner.
    # print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception and not catch_exceptions:
        raise result.exception

    return result
