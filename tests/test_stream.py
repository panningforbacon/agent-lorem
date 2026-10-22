"""Layer 4: GET /runs/{id}/events against a real uvicorn server.

Real server, not TestClient: these tests are about what happens on the socket
(a client hanging up mid-stream, reconnecting, two clients at once), and
TestClient has no socket to hang up.
"""

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Any

import httpx2
import pytest
from _support import StartServer, data, deltas, ids, parse_sse, read_events

from agent_lorem.agent import LOREM_PATH, build_agent
from agent_lorem.server import create_app

LOREM = LOREM_PATH.read_text(encoding="utf-8").strip()
SLOW = 0.01  # ~2 s per run: long enough to interrupt mid-stream

Serve = Callable[..., httpx2.Client]


@pytest.fixture
def serve(start_server: StartServer) -> Iterator[Serve]:
    with ExitStack() as stack:

        def make(*, word_delay_seconds: float = 0) -> httpx2.Client:
            app = create_app(build_agent(word_delay_seconds=word_delay_seconds))
            live = start_server(app)
            return stack.enter_context(
                httpx2.Client(base_url=live.base_url, timeout=10)
            )

        yield make


def _start(client: httpx2.Client) -> dict[str, Any]:
    response = client.post("/runs", json={"userId": "u1", "message": "hi"})
    assert response.status_code == 202, response.text
    return response.json()


def _wait_until_finished(client: httpx2.Client, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} still running")


@pytest.mark.parametrize("when", ["live", "after_completion"])
def test_stream_delivers_every_event_then_end(serve: Serve, when: str) -> None:
    client = serve(word_delay_seconds=SLOW if when == "live" else 0)
    run = _start(client)
    if when == "after_completion":
        _wait_until_finished(client, run["runId"])

    events = read_events(client, run["runId"])

    *adk_events, end = events
    assert ids(events) == list(range(1, len(events) + 1))  # gapless, 1-based
    assert deltas(adk_events) == LOREM
    assert end["event"] == "end"
    assert data(end)["status"] == "completed"
    assert data(end)["eventsSeen"] == len(adk_events)


def test_disconnecting_does_not_cancel_the_run(serve: Serve) -> None:
    client = serve(word_delay_seconds=SLOW)
    run = _start(client)

    read_events(client, run["runId"], stop_after=5)

    assert _wait_until_finished(client, run["runId"])["status"] == "completed"


def test_reconnect_with_last_event_id_resumes_without_gaps(serve: Serve) -> None:
    client = serve(word_delay_seconds=SLOW)
    run = _start(client)

    first = read_events(client, run["runId"], stop_after=10)
    rest = read_events(client, run["runId"], last_event_id=int(first[-1]["id"]))

    assert ids(first + rest) == list(range(1, len(first + rest) + 1))
    assert deltas(first + rest) == LOREM


def test_after_end_reconnect_gets_204(serve: Serve) -> None:
    client = serve()
    run = _start(client)
    _wait_until_finished(client, run["runId"])
    end_id = int(read_events(client, run["runId"])[-1]["id"])

    response = client.get(
        f"/runs/{run['runId']}/events", headers={"Last-Event-ID": str(end_id)}
    )

    assert response.status_code == 204  # what stops EventSource reconnecting


def test_reconnect_that_missed_only_end_gets_end_again(serve: Serve) -> None:
    client = serve()
    run = _start(client)
    _wait_until_finished(client, run["runId"])
    last_adk_id = int(read_events(client, run["runId"])[-2]["id"])

    events = read_events(client, run["runId"], last_event_id=last_adk_id)

    assert [e.get("event") for e in events] == ["end"]


def test_cancel_ends_the_stream_with_aborted(serve: Serve) -> None:
    client = serve(word_delay_seconds=SLOW)
    run = _start(client)
    events: list[dict[str, str]] = []
    with client.stream("GET", f"/runs/{run['runId']}/events") as response:
        for event in parse_sse(response.iter_lines()):
            events.append(event)
            if len(events) == 10:
                client.post(f"/runs/{run['runId']}/cancel")

    end = events[-1]
    assert end["event"] == "end"
    assert data(end)["status"] == "aborted"
    assert data(events[-2])["errorCode"] == "INVOCATION_ABORTED"


def test_two_followers_see_the_same_stream(serve: Serve) -> None:
    client = serve(word_delay_seconds=SLOW)
    run_id = _start(client)["runId"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.map(lambda _: read_events(client, run_id), range(2))

    assert a == b


@pytest.mark.parametrize(
    ("path", "headers", "expected"),
    [
        ("/runs/nope/events", {}, 404),
        ("/runs/{run_id}/events", {"Last-Event-ID": "abc"}, 422),
        ("/runs/{run_id}/events", {"Last-Event-ID": "-1"}, 422),
    ],
)
def test_bad_stream_requests(
    serve: Serve, path: str, headers: dict[str, str], expected: int
) -> None:
    client = serve()
    run_id = _start(client)["runId"]

    response = client.get(path.format(run_id=run_id), headers=headers)

    assert response.status_code == expected
