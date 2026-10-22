"""Layer 5: shutting down while runs are streaming.

The contrast test is the point: stock uvicorn lets a streamed run finish before
it shuts down; DrainingServer aborts it first.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import httpx2
import pytest
from _support import SseEvent, StartServer, data, read_events
from fastapi.testclient import TestClient

from agent_lorem.agent import build_agent
from agent_lorem.server import begin_shutdown, create_app


def _stream_then_shut_down(
    start_server: StartServer, *, draining: bool, word_delay_seconds: float
) -> tuple[list[SseEvent], float]:
    """Open a stream on a running run, shut down mid-run; return what arrived."""
    app = create_app(build_agent(word_delay_seconds=word_delay_seconds))
    live = start_server(app, draining=draining)
    streaming = threading.Event()
    with (
        httpx2.Client(base_url=live.base_url, timeout=30) as client,
        ThreadPoolExecutor(1) as pool,
    ):
        run_id = client.post("/runs", json={"userId": "u1", "message": "hi"}).json()[
            "runId"
        ]

        def mark_streaming(received: list[SseEvent]) -> None:
            if len(received) == 5:
                streaming.set()

        events = pool.submit(read_events, client, run_id, on_event=mark_streaming)
        assert streaming.wait(timeout=10), "stream never started"
        shutdown_seconds = live.stop()
        return events.result(timeout=30), shutdown_seconds


def test_draining_server_aborts_streamed_runs_before_draining(
    start_server: StartServer,
) -> None:
    # 0.05 s/word: the run alone would take ~10 s.
    events, shutdown_seconds = _stream_then_shut_down(
        start_server, draining=True, word_delay_seconds=0.05
    )

    assert shutdown_seconds < 2
    end = events[-1]
    assert end["event"] == "end"  # the stream ended properly, not cut off
    assert data(end)["status"] == "aborted"
    assert data(events[-2])["errorCode"] == "INVOCATION_ABORTED"


def test_stock_uvicorn_waits_for_streamed_runs_to_finish(
    start_server: StartServer,
) -> None:
    # Characterizes the problem DrainingServer exists for. 0.01 s/word keeps
    # the wait to ~2 s; with a real model it has no bound.
    events, _ = _stream_then_shut_down(
        start_server, draining=False, word_delay_seconds=0.01
    )

    # Shut down mid-run, yet the run completed: the drain waited for the open
    # stream, the stream waited for the run, and the lifespan's abort came last.
    assert data(events[-1])["status"] == "completed"


def test_new_runs_are_refused_once_shutdown_begins() -> None:
    app = create_app(build_agent(word_delay_seconds=0.05))
    with TestClient(app) as client:
        running = client.post("/runs", json={"userId": "u1", "message": "hi"}).json()
        assert client.portal is not None
        client.portal.call(begin_shutdown, app)

        refused = client.post("/runs", json={"userId": "u1", "message": "hi"})
        aborted = client.get(f"/runs/{running['runId']}").json()

    assert refused.status_code == 503
    assert aborted["status"] == "aborted"
    # Leaving `with` ran the lifespan's own aclose() a second time: no error.


@pytest.mark.anyio
async def test_begin_shutdown_before_startup_is_a_no_op() -> None:
    # Ctrl+C during startup: the lifespan hasn't created a RunManager yet.
    await begin_shutdown(create_app())
