import asyncio
import contextlib
import logging

import uvicorn

from .cli import run_cli
from .server import create_app
from .serving import DrainingServer


def _configure_logging() -> None:
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("agent_lorem").setLevel(logging.DEBUG)


def main() -> None:
    _configure_logging()
    asyncio.run(run_cli())


def serve() -> None:
    _configure_logging()
    app = create_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        timeout_graceful_shutdown=10,
        # Backstop, not the mechanism: if a connection still won't close this
        # many seconds after runs were aborted (a client that stopped reading,
        # say), uvicorn cancels it rather than waiting forever.
    )
    # An app object rather than an import string, because DrainingServer needs
    # the app itself. The cost: no --reload, which `uvicorn ... --factory`
    # still offers for development, with stock shutdown behavior.
    #
    # After a graceful shutdown uvicorn re-raises the signal it caught, so the
    # process exits "like a Ctrl+C". uvicorn.run() swallows the resulting
    # KeyboardInterrupt; calling server.run() directly, so must we.
    with contextlib.suppress(KeyboardInterrupt):
        DrainingServer(config, app).run()
