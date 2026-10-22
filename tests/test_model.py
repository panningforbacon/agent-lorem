import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from agent_lorem.agent import get_lorem_ipsum
from agent_lorem.model import TOOL_NAME, MockLoremLlm

pytestmark = pytest.mark.anyio

TEXT = "Lorem ipsum\n\ndolor  sit amet."


def _user_turn() -> LlmRequest:
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text="hi")])]
    )


def _after_tool_turn(text: str) -> LlmRequest:
    response = types.FunctionResponse(name=TOOL_NAME, response={"text": text})
    return LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(function_response=response)])
        ]
    )


async def _collect(request: LlmRequest, *, stream: bool) -> list[LlmResponse]:
    model = MockLoremLlm(word_delay_seconds=0)
    return [r async for r in model.generate_content_async(request, stream=stream)]


def _parts(response: LlmResponse) -> list[types.Part]:
    return (response.content and response.content.parts) or []


def test_tool_name_matches_the_real_tool() -> None:
    assert get_lorem_ipsum.__name__ == TOOL_NAME


async def test_first_turn_streams_thoughts_then_a_function_call() -> None:
    *partials, aggregate = await _collect(_user_turn(), stream=True)

    assert all(r.partial for r in partials)
    assert not aggregate.partial
    streamed_thoughts = "".join(
        p.text for r in partials for p in _parts(r) if p.thought and p.text
    )
    thought, call = _parts(aggregate)
    assert thought.text == streamed_thoughts
    assert call.function_call is not None
    assert call.function_call.name == TOOL_NAME


async def test_answer_deltas_reassemble_the_text_exactly() -> None:
    *partials, aggregate = await _collect(_after_tool_turn(TEXT), stream=True)

    deltas = [p.text for r in partials for p in _parts(r) if p.text]
    assert len(deltas) == len(TEXT.split())
    assert "".join(deltas) == TEXT  # whitespace, paragraph break included
    assert not aggregate.partial
    assert _parts(aggregate)[0].text == TEXT


@pytest.mark.parametrize(
    "request_factory", [_user_turn, lambda: _after_tool_turn(TEXT)]
)
async def test_non_streaming_yields_exactly_one_aggregate(request_factory) -> None:
    responses = await _collect(request_factory(), stream=False)

    assert len(responses) == 1
    assert not responses[0].partial
