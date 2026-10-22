"""Layer 6: RunLimits. Run memory stays bounded however long the server lives."""

import time
from dataclasses import dataclass

import pytest
from _support import data, parse_sse
from fastapi.testclient import TestClient
from google.adk import Runner

from agent_lorem.agent import build_agent
from agent_lorem.runner import build_runner, create_session
from agent_lorem.runs import AtCapacityError, Run, RunLimits, RunManager, RunStatus
from agent_lorem.server import create_app

SLOW = 0.01  # ~2 s per run: long enough to still be running when we check


@dataclass
class FakeClock:
    """Time moves only when a test says so: no sleeping, no flakiness."""

    now: float = 0.0

    def __call__(self) -> float:
        return self.now


async def _start(runs: RunManager, runner: Runner) -> Run:
    session = await create_session(runner, user_id="u1")  # fresh session: never busy
    return runs.start(user_id="u1", session_id=session.id, text="hi")


async def _finished(run: Run) -> Run:
    assert run.task is not None
    await run.task
    return run


@pytest.fixture
def slow_runner() -> Runner:
    return build_runner(build_agent(word_delay_seconds=SLOW))


# --- Retention ---------------------------------------------------------------


@pytest.mark.anyio
async def test_finished_runs_expire_after_the_ttl(runner: Runner) -> None:
    clock = FakeClock()
    async with RunManager(runner, RunLimits(finished_ttl_seconds=60), clock) as runs:
        run = await _finished(await _start(runs, runner))

        clock.now = 59.9
        assert runs.get(run.run_id) is run
        clock.now = 60
        assert runs.get(run.run_id) is None


@pytest.mark.anyio
async def test_oldest_finished_runs_go_first_beyond_max_finished(runner: Runner) -> None:
    async with RunManager(runner, RunLimits(max_finished=2)) as runs:
        first, second, third = [await _finished(await _start(runs, runner)) for _ in range(3)]

        assert runs.get(first.run_id) is None
        assert runs.get(second.run_id) is second
        assert runs.get(third.run_id) is third


@pytest.mark.anyio
async def test_running_runs_are_never_evicted(slow_runner: Runner) -> None:
    clock = FakeClock()
    limits = RunLimits(max_finished=0, finished_ttl_seconds=0)
    async with RunManager(slow_runner, limits, clock) as runs:
        run = await _start(runs, slow_runner)

        clock.now = 1e9
        assert runs.get(run.run_id) is run  # still running: kept
        await _finished(run)
        assert runs.get(run.run_id) is None  # finished: gone at once


@pytest.mark.anyio
async def test_a_follower_of_an_evicted_run_still_reads_to_the_end(runner: Runner) -> None:
    clock = FakeClock()
    async with RunManager(runner, RunLimits(finished_ttl_seconds=1), clock) as runs:
        run = await _finished(await _start(runs, runner))
        clock.now = 2
        assert runs.get(run.run_id) is None

        # Eviction drops the registry's reference, not the follower's.
        replayed = [entry async for entry in run.follow()]

    assert len(replayed) == run.events_seen


# --- Capacity ----------------------------------------------------------------


@pytest.mark.anyio
async def test_starts_beyond_max_running_are_refused(slow_runner: Runner) -> None:
    async with RunManager(slow_runner, RunLimits(max_running=1)) as runs:
        first = await _start(runs, slow_runner)
        with pytest.raises(AtCapacityError):
            await _start(runs, slow_runner)

        runs.cancel(first.run_id)
        await _finished(first)
        second = await _start(runs, slow_runner)  # the slot is free again

        assert second.status is RunStatus.RUNNING


# --- Log size ----------------------------------------------------------------


@pytest.mark.anyio
async def test_a_run_whose_log_outgrows_the_limit_is_stopped_and_failed(
    runner: Runner,
) -> None:
    async with RunManager(runner, RunLimits(max_log_bytes=5_000)) as runs:
        run = await _finished(await _start(runs, runner))

    assert run.status is RunStatus.FAILED
    assert run.error == "event log exceeded 5000 bytes"
    assert run.log_bytes > 5_000  # overshoots by the events already in flight
    assert run.events_seen < 200  # far short of a full run
    assert "INVOCATION_ABORTED" in run.log[-1]  # stopped via ADK's own abort


# --- Over HTTP ---------------------------------------------------------------


def test_a_full_server_answers_503_with_retry_after() -> None:
    app = create_app(build_agent(word_delay_seconds=SLOW), RunLimits(max_running=1))
    with TestClient(app) as client:
        client.post("/runs", json={"userId": "u1", "message": "hi"})
        refused = client.post("/runs", json={"userId": "u2", "message": "hi"})

    assert refused.status_code == 503
    assert refused.headers["Retry-After"] == "5"


def test_an_evicted_run_is_404_everywhere() -> None:
    app = create_app(build_agent(word_delay_seconds=0), RunLimits(max_finished=0))
    with TestClient(app) as client:
        run_id = client.post("/runs", json={"userId": "u1", "message": "hi"}).json()["runId"]
        deadline = time.monotonic() + 5
        while client.get(f"/runs/{run_id}").status_code != 404:  # gone once finished
            assert time.monotonic() < deadline, "run was never evicted"
            time.sleep(0.01)

        assert client.get(f"/runs/{run_id}/events").status_code == 404
        assert client.post(f"/runs/{run_id}/cancel").status_code == 404
        assert client.get("/runs", params={"userId": "u1"}).json() == []


def test_the_stream_of_an_overflowing_run_ends_failed() -> None:
    app = create_app(build_agent(word_delay_seconds=0), RunLimits(max_log_bytes=5_000))
    with TestClient(app) as client:
        run_id = client.post("/runs", json={"userId": "u1", "message": "hi"}).json()["runId"]
        with client.stream("GET", f"/runs/{run_id}/events") as response:
            events = list(parse_sse(response.iter_lines()))

    end = data(events[-1])
    assert end["status"] == "failed"
    assert end["error"] == "event log exceeded 5000 bytes"