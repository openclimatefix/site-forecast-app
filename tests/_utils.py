"""Testing utils."""

from typer.testing import CliRunner


def run_typer_script(func, args: list[str], catch_exceptions: bool = False):
    """Util to test typer scripts while showing the stdout."""
    runner = CliRunner()
    result = runner.invoke(func, args, catch_exceptions=catch_exceptions)
    if result.exception and not catch_exceptions:
        raise result.exception
    return result
