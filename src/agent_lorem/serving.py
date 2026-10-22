import socket

import uvicorn
from fastapi import FastAPI

from .server import begin_shutdown


class DrainingServer(uvicorn.Server):
    """uvicorn, but runs are aborted *before* connections are drained.

    uvicorn's shutdown order is: stop accepting, wait for open connections to
    close, then run the lifespan's teardown. An SSE stream closes only when its
    run ends, so with stock uvicorn, Ctrl+C waits for every streamed run to
    finish on its own and the lifespan's abort arrives with nothing to abort.
    Aborting first makes every stream reach its `end` event, so the drain
    finishes at once.

    Both Ctrl+C (handle_exit) and `server.should_exit = True` end up in
    shutdown(), so tests exercise the same path a signal does.
    """

    def __init__(self, config: uvicorn.Config, app: FastAPI) -> None:
        super().__init__(config)
        # config.loaded_app can't be used: uvicorn wraps the app in
        # middleware, so it isn't our FastAPI object and has no .state.
        self._app = app

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        # Overrides a public but undocumented uvicorn method: pin uvicorn and
        # re-check this on upgrades. The stream tests fail loudly if it breaks.
        await begin_shutdown(self._app)
        await super().shutdown(sockets)
