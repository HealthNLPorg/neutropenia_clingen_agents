import click

from ..utils.rest import read_root


@click.command("rest", context_settings={"show_default": True})
@click.option(
    "-h",
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host address to serve the REST app.",
)
@click.option("--message", type=str)
@click.option(
    "-p", "--port", type=int, default=8000, help="Port to serve the REST app."
)
@click.option(
    "--reload",
    type=bool,
    is_flag=True,
    default=False,
    help="Auto-reload the REST app.",
)
def rest_command(message: str, host: str, port: int, reload: bool) -> None:
    """Start a REST application from a model."""
    import uvicorn

    uvicorn.run(read_root(message), host=host, port=port, reload=reload)
