import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from _support import LiveServer, StartServer
from fastapi import FastAPI
from google.adk import Runner

from agent_lorem.agent import build_agent
from agent_lorem.runner import build_runner
from agent_lorem.serving import DrainingServer


@pytest.fixture
def anyio_backend() -> str:
    # AnyIO's pytest plugin runs `@pytest.mark.anyio` tests on this backend.
    # ADK is asyncio-only, so pin it rather than let anyio try trio too.
    return "asyncio"


@pytest.fixture
def runner() -> Runner:
    # delay=0: the stream still yields to the event loop between words
    # (asyncio.sleep(0)), so ordering behaves like production, minus the wait.
    return build_runner(build_agent(word_delay_seconds=0))


@pytest.fixture
def start_server() -> Iterator[StartServer]:
    """Real uvicorn in a background thread, on a port the OS picks."""
    live_servers: list[LiveServer] = []

    def start(app: FastAPI, *, draining: bool = True) -> LiveServer:
        config = uvicorn.Config(app, port=0, log_level="warning")
        server = DrainingServer(config, app) if draining else uvicorn.Server(config)
        # Off the main thread, uvicorn skips installing signal handlers.
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]  # port=0: OS picked
        live = LiveServer(f"http://127.0.0.1:{port}", server, thread)
        live_servers.append(live)
        return live

    yield start
    for live in live_servers:
        if live.thread.is_alive():
            live.stop()
