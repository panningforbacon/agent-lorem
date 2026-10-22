"""FastAPI front end.

Layer 2: runs execute in background tasks; the request only starts them.
Layer 3: an explicit cancel endpoint. Closing a connection never cancels.
Layer 4: an SSE stream per run that replays from Last-Event-ID, then follows.
Layer 5: begin_shutdown() lets the process abort runs before uvicorn drains.
Layer 6: RunLimits bound running runs, retained runs and per-run log size.
Layer 7: GET / serves the browser client, static/index.html.
"""

from collections.abc import AsyncGenerator
from contextlib import aclosing, asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from google.adk import Agent, Runner
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from .agent import build_agent
from .runner import build_runner, create_session, get_session
from .runs import (
    AtCapacityError,
    Run,
    RunFinishedError,
    RunLimits,
    RunManager,
    RunManagerClosedError,
    RunStatus,
    SessionBusyError,
)


class _CamelModel(BaseModel):
    # Same convention as ADK's own API models (google.adk.cli.utils.common):
    # camelCase on the wire to match Event JSON, snake_case accepted on input.
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, from_attributes=True
    )


class RunRequest(_CamelModel):
    user_id: str = Field(min_length=1)
    session_id: str | None = None  # None: start a new session
    message: str = Field(min_length=1)


class RunView(_CamelModel):
    """What clients see of a Run: no task, no abort handle."""

    run_id: str
    user_id: str
    session_id: str
    status: RunStatus
    events_seen: int
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class _State(TypedDict):
    runner: Runner
    runs: RunManager


def _get_runner(request: Request) -> Runner:
    return request.state.runner


def _get_runs(request: Request) -> RunManager:
    return request.state.runs


# Idiom: Annotated dependency aliases; declare `runs: RunsDep` on any route.
RunnerDep = Annotated[Runner, Depends(_get_runner)]
RunsDep = Annotated[RunManager, Depends(_get_runs)]


def _get_run(run_id: str, runs: RunsDep) -> Run:
    # A dependency, not a check inside the handler: dependencies run before a
    # streaming handler starts, so this 404 is a real status code. Raised from
    # inside an SSE generator, the 200 headers would already be on the wire.
    if (run := runs.get(run_id)) is None:
        raise HTTPException(404, f"Run not found: {run_id}")
    return run


RunDep = Annotated[Run, Depends(_get_run)]


def _resume_after(
    run: RunDep,
    # FastAPI maps last_event_id <-> the `Last-Event-ID` header. Browsers
    # send it automatically when an EventSource reconnects.
    last_event_id: Annotated[int, Header(ge=0)] = 0,
) -> int:
    # Sequence numbers: 1..N are ADK events, N+1 is `end`. A client whose
    # Last-Event-ID is already past N has seen `end`: 204 is the one status
    # that tells EventSource to stop reconnecting. Must be a dependency for
    # the same reason as the 404 above. (FastAPI sends a 204 with no body.)
    if run.finished and last_event_id > len(run.log):
        raise HTTPException(204)
    return last_event_id


def _view(run: Run) -> RunView:
    return RunView.model_validate(run)  # from_attributes: read the dataclass


_INDEX_HTML = Path(__file__).with_name("static") / "index.html"

# How long a client should wait before retrying a refused start. A hint, not
# a promise: a mock run takes ~10 s, so a slot usually frees up sooner.
_RETRY_AFTER_SECONDS = "5"


def create_app(agent: Agent | None = None, limits: RunLimits = RunLimits()) -> FastAPI:
    # Factory, not a module-level `app`: each test gets its own app, hence its
    # own InMemorySessionService and RunManager. uvicorn calls it via --factory.
    resolved_agent = agent or build_agent()

    @asynccontextmanager  # Idiom: lifespan = setup before `yield`, teardown after.
    async def lifespan(app: FastAPI) -> AsyncGenerator[_State]:
        # Nested `async with`: exits in reverse order, so every run is aborted
        # and awaited *before* the Runner it uses is closed.
        async with (
            build_runner(resolved_agent) as runner,
            RunManager(runner, limits) as runs,
        ):
            # Two handles to the same RunManager, for two kinds of caller:
            # - lifespan state (yielded) is copied into each request's
            #   request.state: how route handlers reach it.
            # - app.state is how code *outside* any request reaches it:
            #   begin_shutdown(), called by the process, not by a request.
            app.state.runs = runs
            try:
                yield {"runner": runner, "runs": runs}
            finally:
                del app.state.runs

    app = FastAPI(title="agent-lorem", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        # Same origin as the API, so the page needs no CORS configuration.
        return FileResponse(_INDEX_HTML)

    @app.post(
        "/runs", status_code=status.HTTP_202_ACCEPTED, response_model_exclude_none=True
    )
    async def start_run(
        body: RunRequest, response: Response, runner: RunnerDep, runs: RunsDep
    ) -> RunView:
        if body.session_id is None:
            session = await create_session(runner, user_id=body.user_id)
        else:
            session = await get_session(
                runner, user_id=body.user_id, session_id=body.session_id
            )
            if session is None:
                raise HTTPException(404, f"Session not found: {body.session_id}")
        try:
            run = runs.start(
                user_id=body.user_id, session_id=session.id, text=body.message
            )
        except SessionBusyError:
            raise HTTPException(409, f"Session {session.id} already has a running run")
        except RunManagerClosedError:
            raise HTTPException(503, "Server is shutting down")
        except AtCapacityError:
            # 503, not 429: the limit is the server's, not this client's rate.
            # Retry-After tells well-behaved clients when to come back.
            raise HTTPException(
                503,
                f"{limits.max_running} runs already running",
                headers={"Retry-After": _RETRY_AFTER_SECONDS},
            )
        # 202 + Location: "accepted, not done; poll here for the outcome".
        response.headers["Location"] = f"/runs/{run.run_id}"
        return _view(run)

    @app.get("/runs", response_model_exclude_none=True)
    async def list_runs(
        runs: RunsDep, user_id: Annotated[str, Query(alias="userId", min_length=1)]
    ) -> list[RunView]:
        # How a reopened tab finds its runs again. No auth yet: any caller can
        # list any user's runs.
        return [_view(r) for r in runs.list_for_user(user_id)]

    @app.get("/runs/{run_id}", response_model_exclude_none=True)
    async def get_run(run: RunDep) -> RunView:
        return _view(run)

    @app.get("/runs/{run_id}/events", response_class=EventSourceResponse)
    async def stream_run_events(
        # RunDep appears here and inside _resume_after; FastAPI caches
        # dependencies per request, so _get_run still runs once.
        run: RunDep,
        after: Annotated[int, Depends(_resume_after)],
    ) -> AsyncGenerator[ServerSentEvent]:
        # A generator's body runs only when FastAPI pulls the first event,
        # after the 200 headers are sent. Every status decision lives in the
        # dependencies above; from here on, the only way out is the stream.
        async with aclosing(run.follow(after=after)) as entries:
            async for seq, data in entries:
                # raw_data: the log already holds JSON; `data=` would encode it
                # a second time, into a quoted string.
                yield ServerSentEvent(id=str(seq), raw_data=data)
        # Not `data=_view(run)`: FastAPI dumps a model's data without aliases,
        # so the end event would arrive snake_case while everything else is
        # camelCase.
        end = _view(run).model_dump_json(by_alias=True, exclude_none=True)
        yield ServerSentEvent(id=str(len(run.log) + 1), event="end", raw_data=end)

    @app.post(
        "/runs/{run_id}/cancel",
        status_code=status.HTTP_202_ACCEPTED,
        response_model_exclude_none=True,
    )
    async def cancel_run(run_id: str, runs: RunsDep) -> RunView:
        # 202, not 200: the abort is requested, not finished. The returned view
        # still says "running"; poll GET /runs/{run_id} for "aborted".
        try:
            run = runs.cancel(run_id)
        except RunFinishedError:
            raise HTTPException(409, f"Run {run_id} is no longer running")
        if run is None:
            raise HTTPException(404, f"Run not found: {run_id}")
        return _view(run)

    return app


async def begin_shutdown(app: FastAPI) -> None:
    """Abort every run so that every open SSE stream ends.

    For the process to call when shutdown *starts*. uvicorn waits for open
    connections to close before it runs the lifespan's teardown, and a stream
    only closes when its run ends: left to the lifespan alone, shutdown waits
    for every streamed run to finish on its own.
    """
    runs: RunManager | None = getattr(app.state, "runs", None)
    if runs is not None:  # None: shutdown before startup finished
        await runs.aclose()
