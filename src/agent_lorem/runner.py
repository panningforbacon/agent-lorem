import asyncio
from collections.abc import AsyncGenerator

from google.adk import Agent, Event, Runner
from google.adk.agents.run_config import (
    RunConfig,
    StreamingMode,  # pyright: ignore[reportPrivateImportUsage]
)
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types

APP_NAME = "agent_lorem"

_STREAMING = RunConfig(streaming_mode=StreamingMode.SSE)


def build_runner(agent: Agent) -> Runner:
    return Runner(
        app_name=APP_NAME, agent=agent, session_service=InMemorySessionService()
    )


async def create_session(runner: Runner, *, user_id: str) -> Session:
    return await runner.session_service.create_session(
        app_name=runner.app_name, user_id=user_id
    )


async def get_session(
    runner: Runner, *, user_id: str, session_id: str
) -> Session | None:
    return await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session_id
    )


def run_turn(
    runner: Runner,
    *,
    user_id: str,
    session_id: str,
    text: str,
    abort_signal: asyncio.Event | None = None,
) -> AsyncGenerator[Event]:
    return runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
        run_config=_STREAMING,
        abort_signal=abort_signal,
    )
