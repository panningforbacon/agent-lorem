import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from functools import partial
from typing import Any

import pytest
from fastapi.testclient import TestClient
from google.adk import Agent, Event

from agent_lorem.agent import build_agent
from agent_lorem.model import MockLoremLlm
from agent_lorem.runner import get_session
from agent_lorem.runs import RunManager
from agent_lorem.server import create_app

# 200 words at 10 ms: ~2 s per run. Long enough to cancel mid-stream reliably,
# short enough that a missed cancel fails fast instead of hanging.
SLOW = 0.01

MakeClient = Callable[..., TestClient]


@pytest.fixture
def make_client() -> Iterator[MakeClient]:
    # TestClient, not httpx's ASGITransport: ASGITransport never sends the
    # lifespan events, so the Runner and RunManager would never exist.
    # ExitStack shuts down every client a test made, in reverse order.
    with ExitStack() as stack:

        def make(
            *, word_delay_seconds: float = 0, agent: Agent | None = None
        ) -> TestClient:
            app = create_app(
                agent or build_agent(word_delay_seconds=word_delay_seconds)
            )
            return stack.enter_context(TestClient(app))

        yield make


@pytest.fixture
def client(make_client: MakeClient) -> TestClient:
    return make_client()


def _start(client: TestClient, **body: str) -> dict[str, Any]:
    response = client.post("/runs", json={"message": "hi", **body})
    assert response.status_code == 202, response.text
    return response.json()


def _wait_until_finished(
    client: TestClient, run_id: str, timeout: float = 5
) -> dict[str, Any]:
    # Polling is the only way to observe a detached run until Layer 4's stream.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.01)
    raise AssertionError(f"run {run_id} still running after {timeout}s")


def _session_events(client: TestClient, user_id: str, session_id: str) -> list[Event]:
    # The app runs on TestClient's own event loop in a background thread; the
    # portal runs our coroutine on *that* loop instead of a second one.
    assert client.portal is not None
    session = client.portal.call(
        partial(
            get_session,
            client.app_state["runner"],
            user_id=user_id,
            session_id=session_id,
        )
    )
    assert session is not None
    return session.events


# --- Layer 2: detached runs -------------------------------------------------


def test_start_returns_202_before_the_run_finishes(make_client: MakeClient) -> None:
    client = make_client(word_delay_seconds=SLOW)
    response = client.post("/runs", json={"userId": "u1", "message": "hi"})

    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "running"
    assert response.headers["Location"] == f"/runs/{run['runId']}"
    assert "finishedAt" not in run  # exclude_none: not finished, so absent


def test_run_completes_in_the_background(client: TestClient) -> None:
    run = _start(client, userId="u1")

    finished = _wait_until_finished(client, run["runId"])

    assert finished["status"] == "completed"
    assert finished["eventsSeen"] > 200  # thoughts, call, response, 200 deltas, answer
    events = _session_events(client, "u1", run["sessionId"])
    assert len(events) == 4 and events[-1].is_final_response()


def test_follow_up_run_appends_to_the_same_session(client: TestClient) -> None:
    first = _start(client, userId="u1")
    _wait_until_finished(client, first["runId"])
    second = _start(client, userId="u1", sessionId=first["sessionId"])
    _wait_until_finished(client, second["runId"])

    assert len(_session_events(client, "u1", first["sessionId"])) == 8


def test_second_run_on_a_busy_session_is_409(make_client: MakeClient) -> None:
    client = make_client(word_delay_seconds=SLOW)
    first = _start(client, userId="u1")

    response = client.post(
        "/runs", json={"userId": "u1", "sessionId": first["sessionId"], "message": "hi"}
    )

    assert response.status_code == 409


def test_runs_are_listed_per_user_in_creation_order(client: TestClient) -> None:
    a = _start(client, userId="u1")
    _start(client, userId="u2")
    b = _start(client, userId="u1")

    listed = client.get("/runs", params={"userId": "u1"}).json()

    assert [r["runId"] for r in listed] == [a["runId"], b["runId"]]


def test_failing_tool_marks_the_run_failed(make_client: MakeClient) -> None:
    def get_lorem_ipsum() -> dict:  # same name the mock model calls
        raise RuntimeError("tool exploded")

    agent = Agent(
        name="lorem_agent",
        model=MockLoremLlm(word_delay_seconds=0),
        tools=[get_lorem_ipsum],
    )
    client = make_client(agent=agent)

    finished = _wait_until_finished(client, _start(client, userId="u1")["runId"])

    assert finished["status"] == "failed"
    assert "tool exploded" in finished["error"]


@pytest.mark.parametrize(
    ("method", "path"), [("get", "/runs/nope"), ("post", "/runs/nope/cancel")]
)
def test_unknown_run_is_404(client: TestClient, method: str, path: str) -> None:
    assert client.request(method, path).status_code == 404


def test_unknown_or_foreign_session_is_404(client: TestClient) -> None:
    owner = _start(client, userId="u1")
    for user, session_id in [("u1", "nope"), ("u2", owner["sessionId"])]:
        response = client.post(
            "/runs", json={"userId": user, "sessionId": session_id, "message": "hi"}
        )
        assert response.status_code == 404


def test_each_app_has_its_own_sessions_and_runs(make_client: MakeClient) -> None:
    first, other = make_client(), make_client()
    run = _start(first, userId="u1")

    assert other.get(f"/runs/{run['runId']}").status_code == 404
    response = other.post(
        "/runs", json={"userId": "u1", "sessionId": run["sessionId"], "message": "hi"}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"message": "hi"},
        {"userId": "", "message": "hi"},
        {"userId": "u1", "message": ""},
    ],
)
def test_invalid_request_is_422(client: TestClient, body: dict[str, str]) -> None:
    assert client.post("/runs", json=body).status_code == 422


def test_listing_without_user_is_422(client: TestClient) -> None:
    assert client.get("/runs").status_code == 422


# --- Layer 3: cancel ---------------------------------------------------------


def test_cancel_aborts_a_running_run(make_client: MakeClient) -> None:
    client = make_client(word_delay_seconds=SLOW)
    run = _start(client, userId="u1")

    response = client.post(f"/runs/{run['runId']}/cancel")

    assert response.status_code == 202
    assert response.json()["status"] == "running"  # requested, not yet done
    finished = _wait_until_finished(client, run["runId"])
    assert finished["status"] == "aborted"
    assert finished["eventsSeen"] < 200
    events = _session_events(client, "u1", run["sessionId"])
    assert events[-1].error_code == "INVOCATION_ABORTED"
    # is_final_response() is also True for the *user's* message event, so
    # filter by author: no answer from the agent made it into the session.
    assert not any(
        e.author == "lorem_agent" and e.is_final_response() and e.content
        for e in events
    )


def test_cancel_is_idempotent_while_running(make_client: MakeClient) -> None:
    client = make_client(word_delay_seconds=SLOW)
    run_id = _start(client, userId="u1")["runId"]

    codes = [client.post(f"/runs/{run_id}/cancel").status_code for _ in range(2)]

    # The second call usually lands before ADK has noticed the first.
    assert codes[0] == 202 and codes[1] in (202, 409)
    assert _wait_until_finished(client, run_id)["status"] == "aborted"


def test_cancel_after_completion_is_409(client: TestClient) -> None:
    run_id = _start(client, userId="u1")["runId"]
    _wait_until_finished(client, run_id)

    assert client.post(f"/runs/{run_id}/cancel").status_code == 409


def test_session_is_free_again_after_cancel(make_client: MakeClient) -> None:
    client = make_client(word_delay_seconds=SLOW)
    first = _start(client, userId="u1")
    client.post(f"/runs/{first['runId']}/cancel")
    _wait_until_finished(client, first["runId"])

    second = _start(client, userId="u1", sessionId=first["sessionId"])

    assert second["status"] == "running"


# --- Shutdown -----------------------------------------------------------------


def test_app_shutdown_aborts_running_runs() -> None:
    # 0.05 s/word: a full run would take ~10 s, so a fast exit proves the abort.
    with TestClient(create_app(build_agent(word_delay_seconds=0.05))) as client:
        run_id = _start(client, userId="u1")["runId"]
        runs: RunManager = client.app_state["runs"]
        started = time.monotonic()
    shutdown_seconds = time.monotonic() - started

    run = runs.get(run_id)
    assert run is not None and run.status == "aborted"
    assert shutdown_seconds < 2
