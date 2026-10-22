import asyncio

import pytest
from google.adk import Event, Runner

from agent_lorem.agent import LOREM_PATH
from agent_lorem.runner import create_session, run_turn

pytestmark = pytest.mark.anyio

USER = "test_user"


def _answer_text(event: Event) -> str:
    parts = (event.content and event.content.parts) or []
    return "".join(p.text for p in parts if p.text and not p.thought)


async def _persisted(runner: Runner, session_id: str) -> list[Event]:
    session = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=USER, session_id=session_id
    )
    assert session is not None
    return session.events


async def test_full_turn_streams_deltas_that_match_the_final_answer(
    runner: Runner,
) -> None:
    session = await create_session(runner, user_id=USER)

    events = [
        e
        async for e in run_turn(runner, user_id=USER, session_id=session.id, text="hi")
    ]

    deltas = [_answer_text(e) for e in events if e.partial and _answer_text(e)]
    finals = [_answer_text(e) for e in events if e.is_final_response()]
    expected = LOREM_PATH.read_text(encoding="utf-8").strip()
    assert "".join(deltas) == expected
    assert finals == [expected]


async def test_session_persists_aggregates_never_partials(runner: Runner) -> None:
    session = await create_session(runner, user_id=USER)
    async for _ in run_turn(runner, user_id=USER, session_id=session.id, text="hi"):
        pass

    stored = await _persisted(runner, session.id)

    assert not any(e.partial for e in stored)
    # user message, model's function call, tool's response, model's answer
    assert [e.author for e in stored] == [
        "user",
        "lorem_agent",
        "lorem_agent",
        "lorem_agent",
    ]
    assert stored[1].get_function_calls()
    assert stored[2].get_function_responses()
    assert stored[3].is_final_response()


async def test_abort_mid_stream_ends_without_a_final_answer(runner: Runner) -> None:
    session = await create_session(runner, user_id=USER)
    abort = asyncio.Event()
    deltas_seen = 0
    events: list[Event] = []

    async for event in run_turn(
        runner, user_id=USER, session_id=session.id, text="hi", abort_signal=abort
    ):
        events.append(event)
        if event.partial and _answer_text(event):
            deltas_seen += 1
            if deltas_seen == 3:
                abort.set()

    assert events[-1].error_code == "INVOCATION_ABORTED"
    assert not any(e.is_final_response() and _answer_text(e) for e in events)
    stored = await _persisted(runner, session.id)
    assert stored[-1].error_code == "INVOCATION_ABORTED"
    # Not `== 3`: partial events skip ADK's backpressure (InvocationContext.
    # _enqueue_event), so a few may already be queued when abort lands. Only
    # "the stream stopped early" is a contract; the exact count is scheduling.
    total_words = len(LOREM_PATH.read_text(encoding="utf-8").split())
    assert deltas_seen < total_words
