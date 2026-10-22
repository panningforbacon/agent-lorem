"""Layer 7: the browser client, driven in real Chromium against real uvicorn.

These cover what only a browser does: EventSource's own reconnect rules,
localStorage across reloads and tabs, and rendering raw ADK events without
showing any text twice.

Needs a Chromium that Playwright can launch: `uv run playwright install chromium`
once, or point PLAYWRIGHT_CHROMIUM_EXECUTABLE at an existing build. Without
one, these tests skip rather than fail.
"""

import os
from collections.abc import Iterator

import httpx2
import pytest
from _support import StartServer
from playwright.sync_api import Browser, Error, Page, Request, sync_playwright

from agent_lorem.agent import LOREM_PATH, build_agent
from agent_lorem.model import THOUGHT_CHUNKS
from agent_lorem.runs import RunLimits
from agent_lorem.server import create_app

LOREM = LOREM_PATH.read_text(encoding="utf-8").strip()
SLOW = 0.01  # ~2 s per run: long enough to click Stop or reload mid-stream


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(executable_path=executable)
        except Error as exc:
            pytest.skip(
                f"no launchable Chromium ({exc.message.splitlines()[0]}); "
                "run `uv run playwright install chromium`"
            )
        yield browser
        browser.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    # A fresh context per test: its own localStorage, like a fresh browser profile.
    context = browser.new_context()
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    yield page
    context.close()
    assert not errors, f"JavaScript errors: {errors}"


def _serve(start_server: StartServer, limits: RunLimits = RunLimits()) -> str:
    app = create_app(build_agent(word_delay_seconds=SLOW), limits)
    return start_server(app).base_url


def _wait_for_status(page: Page, state: str) -> None:
    page.wait_for_selector(f'#status[data-state="{state}"]', timeout=15_000)


def _answer(page: Page) -> str:
    return page.locator(".transcript .answer").text_content() or ""


def test_a_full_run_renders_once_and_the_stream_is_closed(
    start_server: StartServer, page: Page
) -> None:
    base_url = _serve(start_server)
    stream_requests: list[str] = []

    def record_stream_request(request: Request) -> None:
        if "/events" in request.url:
            stream_requests.append(request.url)

    page.on("request", record_stream_request)
    page.goto(base_url)

    page.click("#start")
    _wait_for_status(page, "completed")
    page.wait_for_timeout(500)  # time for a wrongful reconnect to show up

    assert (
        _answer(page) == LOREM
    )  # partials appended, aggregate replaced, never doubled
    assert page.locator(".transcript .thought").text_content() == "".join(
        THOUGHT_CHUNKS
    )
    assert page.locator(".tick").count() == 207 + 1  # every event, plus `end`
    assert page.evaluate("source.readyState") == 2  # EventSource.CLOSED
    assert (
        len(stream_requests) == 1
    )  # es.close() on `end`: no reconnect, no 204 round trip


def test_stop_ends_the_run_and_keeps_what_arrived(
    start_server: StartServer, page: Page
) -> None:
    page.goto(_serve(start_server))

    page.click("#start")
    page.wait_for_function("document.querySelectorAll('.tick.answer').length >= 5")
    page.click("#stop")
    _wait_for_status(page, "aborted")

    shown = _answer(page)
    assert 0 < len(shown) < len(LOREM)
    assert LOREM.startswith(shown)  # a clean prefix: nothing lost or garbled
    assert page.get_by_text("The run stopped here.").is_visible()
    assert page.locator("#status").text_content() == "Stopped"


@pytest.mark.parametrize("how", ["reload", "new_tab"])
def test_coming_back_mid_run_replays_and_finishes_without_duplicates(
    start_server: StartServer, page: Page, how: str
) -> None:
    page.goto(_serve(start_server))
    page.click("#start")
    page.wait_for_function("document.querySelectorAll('.tick.answer').length >= 20")

    if how == "reload":
        page.reload()
        again = page
    else:
        # Same context = same localStorage, as when a user reopens the site.
        again = page.context.new_page()
        url = page.url
        page.close()
        again.goto(url)
    _wait_for_status(again, "completed")

    assert _answer(again) == LOREM
    assert again.locator(".transcript .thought").count() == 1


def test_an_evicted_run_is_forgotten_on_return(
    start_server: StartServer, page: Page
) -> None:
    # max_finished=0: a run is evicted the moment it finishes.
    page.goto(_serve(start_server, RunLimits(max_finished=0)))
    page.click("#start")
    _wait_for_status(page, "completed")  # the stream still finishes: it holds the Run

    page.reload()

    page.wait_for_selector("text=Your last run is no longer on the server")
    assert page.locator("#status").text_content() == "Idle"
    assert page.evaluate("localStorage.getItem('runId')") is None


def test_a_full_server_says_when_to_retry(
    start_server: StartServer, browser: Browser
) -> None:
    base_url = _serve(start_server, RunLimits(max_running=1))
    first, second = browser.new_page(), browser.new_page()  # separate contexts
    try:
        first.goto(base_url)
        first.click("#start")
        _wait_for_status(first, "running")
        second.goto(base_url)

        second.click("#start")

        second.wait_for_selector("text=Try again in 5 seconds")
        assert second.locator("#status").text_content() == "Idle"
    finally:
        first.context.close()
        second.context.close()


def test_continue_the_last_session_reuses_it(
    start_server: StartServer, page: Page
) -> None:
    base_url = _serve(start_server)
    page.goto(base_url)
    page.click("#start")
    _wait_for_status(page, "completed")

    page.check("#continue-session")
    page.click("#start")
    _wait_for_status(page, "completed")

    runs = httpx2.get(f"{base_url}/runs", params={"userId": "demo_user"}).json()
    assert len(runs) == 2
    assert runs[0]["sessionId"] == runs[1]["sessionId"]
