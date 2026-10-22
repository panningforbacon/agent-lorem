import asyncio
import contextlib
import signal
import time
from collections.abc import Generator

from google.genai import types

from .agent import build_agent
from .runner import build_runner, create_session, run_turn


@contextlib.contextmanager  # Idiom: context manager pairs install/restore of global state.
def _abort_on_first_ctrl_c(abort: asyncio.Event) -> Generator[None]:
    loop = asyncio.get_running_loop()

    def on_sigint(signum: int, frame: object) -> None:
        loop.call_soon_threadsafe(abort.set)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    previous = signal.signal(signal.SIGINT, on_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


async def run_cli() -> None:
    user_id = "cli_user"
    runner = build_runner(build_agent())
    session = await create_session(runner, user_id=user_id)

    abort = asyncio.Event()
    deltas: list[str] = []
    delta_times: list[float] = []
    final_text: str | None = None
    tool_text: str | None = None

    with _abort_on_first_ctrl_c(abort):
        async for event in run_turn(
            runner,
            user_id=user_id,
            session_id=session.id,
            text="anything",
            abort_signal=abort,
        ):
            if event.error_code:
                print(f"\n[event] {event.error_code}")
            for part in (event.content and event.content.parts) or ():
                match part:
                    case types.Part(thought=True, text=str(t)) if event.partial:
                        print(f"\033[2m{t}\033[0m", end="", flush=True)
                    case types.Part(function_call=types.FunctionCall() as fc) if (
                        not event.partial
                    ):
                        print(f"\n[tool call] {fc.name}({fc.args})")
                    # Mapping sub-pattern rejects response=None and binds text as str.
                    case types.Part(
                        function_response=types.FunctionResponse(
                            response={"text": str(text)}
                        ) as fr
                    ):
                        tool_text = text
                        print(f"[tool result] {fr.name} -> {len(text.split())} words")
                    case types.Part(text=str(t)) if not part.thought and event.partial:
                        deltas.append(t)
                        delta_times.append(time.monotonic())
                        print(t, end="", flush=True)
                    case types.Part(text=str(t)) if (
                        not part.thought and event.is_final_response()
                    ):
                        final_text = t

    print()
    _print_stats(deltas, delta_times, final_text, tool_text, aborted=abort.is_set())

    stored = await runner.session_service.get_session(
        app_name=runner.app_name, user_id=user_id, session_id=session.id
    )
    if stored is not None:
        # Partials are never persisted; on abort the answer aggregate never exists.
        print(f"[session] {len(stored.events)} events persisted")


def _print_stats(
    deltas: list[str],
    times: list[float],
    final_text: str | None,
    tool_text: str | None,
    aborted: bool,
) -> None:
    gaps = [b - a for a, b in zip(times, times[1:])]
    if gaps:
        print(f"[stats] {len(deltas)} deltas, mean gap {sum(gaps) / len(gaps):.3f}s")
    if aborted:
        print(f"[stats] aborted; final aggregate received: {final_text is not None}")
    else:
        matches = final_text == "".join(deltas) == tool_text
        print(f"[stats] final aggregate == concatenated deltas == file text: {matches}")
