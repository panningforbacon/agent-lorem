"""Runs that outlive the HTTP request that started them.

A Run is one agent turn executing in its own asyncio Task, plus an append-only
log of its events that any number of clients can replay and follow. RunManager
is the registry: the only place that knows which runs exist, and the only
handle for stopping one. Single process only; see the multi-worker notes.

RunLimits bound what the registry can hold. Worst case, run memory is
(max_running + max_finished) * max_log_bytes, plus Python's per-object
overhead. ADK sessions are not covered: InMemorySessionService keeps every
session forever.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import TracebackType
from typing import Self

from google.adk import Event, Runner

from .runner import run_turn

logger = logging.getLogger(__name__)

# ADK's marker event when abort_signal stops an invocation.
_ABORTED_CODE = "INVOCATION_ABORTED"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(frozen=True)
class RunLimits:
    max_running: int = 16
    # Finished runs are kept so clients can replay them, then evicted when
    # either limit is hit, oldest-finished first.
    max_finished: int = 256
    finished_ttl_seconds: float = 15 * 60
    # One run's event log. A runaway agent is stopped and marked failed.
    max_log_bytes: int = 1024 * 1024


@dataclass
class Run:
    user_id: str
    session_id: str
    # uuid7 (new in 3.14) is time-ordered: sorting run_ids sorts by creation.
    run_id: str = field(default_factory=lambda: str(uuid.uuid7()))
    status: RunStatus = RunStatus.RUNNING
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    abort: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    # Append-only. Entry i has sequence number i + 1, which is the SSE `id:`.
    # ADK's own Event.id can't serve: partial chunks of one response share it.
    log: list[str] = field(default_factory=list, init=False, repr=False)
    log_bytes: int = field(default=0, init=False)
    # Swapped for a fresh Event on every change; see _wake_followers.
    _changed: asyncio.Event = field(
        default_factory=asyncio.Event, init=False, repr=False
    )

    @property
    def events_seen(self) -> int:
        return len(self.log)

    @property
    def finished(self) -> bool:
        return self.status is not RunStatus.RUNNING

    def record(self, event: Event) -> None:
        # Snapshot as JSON now: ADK builds events with shallow model_copy, so a
        # stored object could share state ADK touches later. JSON also means
        # serializing once, however many clients are following.
        entry = event.model_dump_json(exclude_none=True, by_alias=True)
        self.log.append(entry)
        self.log_bytes += len(entry.encode())  # bytes, not characters
        self._wake_followers()

    def finish(self, status: RunStatus, error: str | None = None) -> None:
        # The only way out of RUNNING, and it always wakes followers: a status
        # change nobody hears about would leave streams waiting forever.
        self.status, self.error = status, error
        self.finished_at = datetime.now(UTC)
        self._wake_followers()

    def _wake_followers(self) -> None:
        # Idiom: broadcast by replacing an Event. Everyone waiting on the old
        # one wakes; later waiters get the new, unset one. Plain sync code, so
        # the producer never awaits (unlike Condition.notify_all, which needs
        # the lock, i.e. an `await`, even inside a `finally` after cancel).
        changed, self._changed = self._changed, asyncio.Event()
        changed.set()

    async def follow(self, after: int = 0) -> AsyncGenerator[tuple[int, str]]:
        """Yield (seq, json) for every entry after `after`: replay, then live."""
        seq = after
        while True:
            while seq < len(self.log):
                seq += 1
                yield seq, self.log[seq - 1]
            if self.finished:
                return
            # No `await` between the checks above and reading self._changed,
            # so the producer can't append-and-wake in the gap: no lost wake-up.
            await self._changed.wait()


class SessionBusyError(Exception):
    """The session already has a running run; two would interleave its history."""


class RunFinishedError(Exception):
    """Cancel was requested for a run that is no longer running."""


class RunManagerClosedError(Exception):
    """The process is shutting down; no new runs."""


class AtCapacityError(Exception):
    """RunLimits.max_running runs are already running."""


class RunManager:
    def __init__(
        self,
        runner: Runner,
        limits: RunLimits = RunLimits(),
        # Injectable so tests can move time forward instead of sleeping.
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runner = runner
        self._limits = limits
        self._clock = clock
        # Holding Run (and so its Task) here is load-bearing: the event loop
        # keeps only *weak* references to tasks, so an unreferenced task can be
        # garbage-collected mid-run. See the asyncio.create_task docs.
        self._runs: dict[str, Run] = {}
        # (expires_at, run_id) in the order runs finished. Every run gets the
        # same TTL, so finish order is expiry order: the oldest is always at
        # the left, and eviction never has to scan.
        self._finished: deque[tuple[float, str]] = deque()
        self._closing = False

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def running_count(self) -> int:
        return len(self._runs) - len(self._finished)

    def start(self, *, user_id: str, session_id: str, text: str) -> Run:
        # Deliberately sync: no `await` between the checks and registering the
        # run, so two concurrent requests can't both pass a check.
        if self._closing:
            # Shutdown aborts runs *before* uvicorn stops accepting
            # connections, so requests can still arrive; turn them away.
            raise RunManagerClosedError
        if self.running_count >= self._limits.max_running:
            raise AtCapacityError
        if any(
            r.session_id == session_id and r.status is RunStatus.RUNNING
            for r in self._runs.values()
        ):
            raise SessionBusyError(session_id)
        run = Run(user_id=user_id, session_id=session_id)
        run.task = asyncio.create_task(self._drive(run, text), name=f"run-{run.run_id}")
        self._runs[run.run_id] = run
        return run

    def get(self, run_id: str) -> Run | None:
        self._evict_expired()
        return self._runs.get(run_id)

    def list_for_user(self, user_id: str) -> list[Run]:
        self._evict_expired()
        return sorted(
            (r for r in self._runs.values() if r.user_id == user_id),
            key=lambda r: r.run_id,  # uuid7: creation order
        )

    def cancel(self, run_id: str) -> Run | None:
        run = self.get(run_id)
        if run is None:
            return None
        if run.status is not RunStatus.RUNNING:
            raise RunFinishedError(run_id)
        # A request, not an act: ADK notices at its next checkpoint, writes
        # INVOCATION_ABORTED, and _drive records the outcome. Idempotent.
        run.abort.set()
        return run

    async def aclose(self, timeout: float = 5.0) -> None:
        """Refuse new runs, abort running ones, hard-cancel any that ignore it.

        Idempotent: the process calls it when shutdown begins, and the
        lifespan calls it again on exit, by which time there's nothing left.
        """
        self._closing = True
        tasks = [r.task for r in self._runs.values() if r.task and not r.task.done()]
        if not tasks:
            return
        for run in self._runs.values():
            run.abort.set()
        _, stragglers = await asyncio.wait(tasks, timeout=timeout)
        for task in stragglers:
            task.cancel()
        await asyncio.gather(*stragglers, return_exceptions=True)

    def _evict_expired(self) -> None:
        # Lazy, no background sweeper: runs on every read and on every finish.
        # Memory only grows when runs finish, so evicting then bounds it; the
        # reads make sure an expired run is never *served*.
        now = self._clock()
        while self._finished and (
            self._finished[0][0] <= now
            or len(self._finished) > self._limits.max_finished
        ):
            _, run_id = self._finished.popleft()
            # A client mid-replay keeps its own reference to the Run, so its
            # stream finishes normally; only new lookups get 404.
            del self._runs[run_id]

    def _retire(self, run: Run) -> None:
        self._finished.append(
            (self._clock() + self._limits.finished_ttl_seconds, run.run_id)
        )
        self._evict_expired()

    async def _drive(self, run: Run, text: str) -> None:
        # Every outcome is caught and recorded, so the task itself never ends
        # with an exception nobody retrieves.
        aborted = overflowed = False
        status, error = RunStatus.FAILED, None
        try:
            async for event in run_turn(
                self._runner,
                user_id=run.user_id,
                session_id=run.session_id,
                text=text,
                abort_signal=run.abort,
            ):
                run.record(event)
                aborted = aborted or event.error_code == _ABORTED_CODE
                if not overflowed and run.log_bytes > self._limits.max_log_bytes:
                    # Stop it through the normal abort path so ADK still writes
                    # its marker; the few events already queued still land.
                    overflowed = True
                    run.abort.set()
            if overflowed:
                error = f"event log exceeded {self._limits.max_log_bytes} bytes"
            else:
                # Status from ADK's own marker, not from run.abort: an abort
                # that lands after the last event changes nothing, and saying
                # "aborted" about a turn that finished would be a lie.
                status = RunStatus.ABORTED if aborted else RunStatus.COMPLETED
        except asyncio.CancelledError:
            # Hard cancel (aclose timeout): ADK never got to write its marker.
            status = RunStatus.ABORTED
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("run %s failed", run.run_id)
        finally:
            run.finish(status, error)
            self._retire(run)
            logger.info(
                "run %s %s after %d events", run.run_id, run.status, run.events_seen
            )
