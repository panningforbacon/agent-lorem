"""Helpers shared by the stream and shutdown tests (fixtures live in conftest).

Plain module, not conftest: pytest warns against importing conftest directly.
"""

import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx2
import uvicorn

SseEvent = dict[str, str]


@dataclass
class LiveServer:
    base_url: str
    server: uvicorn.Server
    thread: threading.Thread

    def stop(self, timeout: float = 30) -> float:
        """Shut down the way Ctrl+C does; return how long it took."""
        started = time.monotonic()
        # handle_exit (the signal handler) does exactly this; shutdown() follows.
        self.server.should_exit = True
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise AssertionError(f"server still running after {timeout}s")
        return time.monotonic() - started


StartServer = Callable[..., LiveServer]


def parse_sse(lines: Iterator[str]) -> Iterator[SseEvent]:
    # Minimal SSE parser: fields until a blank line; `:` lines are comments
    # (keep-alive pings). Our data is single-line JSON, so no multi-line join.
    event: SseEvent = {}
    for line in lines:
        if not line:
            if event:
                yield event
                event = {}
        elif not line.startswith(":"):
            name, _, value = line.partition(":")
            event[name] = value.removeprefix(" ")


def read_events(
    client: httpx2.Client,
    run_id: str,
    *,
    last_event_id: int | None = None,
    stop_after: int | None = None,
    on_event: Callable[[list[SseEvent]], None] | None = None,
) -> list[SseEvent]:
    """Read a run's stream to its end, or until `stop_after` events."""
    headers = {} if last_event_id is None else {"Last-Event-ID": str(last_event_id)}
    received: list[SseEvent] = []
    with client.stream("GET", f"/runs/{run_id}/events", headers=headers) as response:
        assert response.status_code == 200, response.read()
        for event in parse_sse(response.iter_lines()):
            received.append(event)
            if on_event is not None:
                on_event(received)
            if len(received) == stop_after:
                break  # leaving the `with` closes the connection mid-stream
    return received


def data(event: SseEvent) -> dict:
    return json.loads(event["data"])


def answer_text(event: SseEvent) -> str:
    parts = data(event).get("content", {}).get("parts", [])
    return "".join(p["text"] for p in parts if "text" in p and not p.get("thought"))


def deltas(events: list[SseEvent]) -> str:
    adk = [e for e in events if "event" not in e]  # unnamed = ADK events
    return "".join(answer_text(e) for e in adk if data(e).get("partial"))


def ids(events: list[SseEvent]) -> list[int]:
    return [int(e["id"]) for e in events]
