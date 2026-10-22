import asyncio
import logging
import re
from collections.abc import AsyncGenerator
from typing import override

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

WORD_DELAY_SECONDS = 0.05

_WORD_WITH_LEADING_SPACE = re.compile(r"\s*\S+")

THOUGHT_CHUNKS = (
    "The user sent a message. ",
    "Content is irrelevant; the answer is always the same text. ",
    "I should fetch it with get_lorem_ipsum before replying.",
)
TOOL_NAME = "get_lorem_ipsum"

logger = logging.getLogger(__name__)


class MockLoremLlm(BaseLlm):
    """Stateless fake LLM honoring BaseLlm's SSE contract.

    Streaming: N partial deltas, then one partial=False aggregate identical to
    what stream=False would return. That's the contract ADK relies on: only the
    aggregate is persisted to the session and only its function calls execute.
    """

    model: str = "mock-lorem"
    word_delay_seconds: float = WORD_DELAY_SECONDS

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        tool_result = _latest_tool_result(llm_request)
        turn = (
            self._answer_turn(tool_result["text"], stream)
            if tool_result is not None
            else self._thinking_and_tool_call_turn(stream)
        )
        async for response in turn:
            yield response

    async def _thinking_and_tool_call_turn(
        self, stream: bool
    ) -> AsyncGenerator[LlmResponse]:
        call = types.Part(function_call=types.FunctionCall(name=TOOL_NAME, args={}))
        if stream:
            for chunk in THOUGHT_CHUNKS:
                await asyncio.sleep(self.word_delay_seconds)
                yield _model_response(
                    types.Part(text=chunk, thought=True), partial=True
                )
            yield _model_response(call, partial=True)
        yield _model_response(
            types.Part(text="".join(THOUGHT_CHUNKS), thought=True), call, partial=False
        )

    async def _answer_turn(
        self, text: str, stream: bool
    ) -> AsyncGenerator[LlmResponse]:
        chunks = _WORD_WITH_LEADING_SPACE.findall(text)
        sent = 0
        try:
            if stream:
                for chunk in chunks:
                    await asyncio.sleep(self.word_delay_seconds)
                    yield _model_response(types.Part(text=chunk), partial=True)
                    sent += 1
            yield _model_response(types.Part(text=text), partial=False)
        finally:
            # Fires on abort_signal, consumer `break`, or task cancellation:
            # ADK cancels the root task, CancelledError lands in the sleep above.
            if stream and sent < len(chunks):
                logger.info("stream closed after %d/%d words", sent, len(chunks))


def _model_response(*parts: types.Part, partial: bool) -> LlmResponse:
    usage = (
        None
        if partial
        else types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            candidates_token_count=0,
            total_token_count=0,
        )
    )
    return LlmResponse(
        content=types.Content(role="model", parts=list(parts)),
        partial=partial,
        usage_metadata=usage,
    )


def _latest_tool_result(llm_request: LlmRequest) -> dict | None:
    if not llm_request.contents:
        return None
    for part in llm_request.contents[-1].parts or ():
        if (fr := part.function_response) and fr.name == TOOL_NAME:
            return fr.response
    return None
