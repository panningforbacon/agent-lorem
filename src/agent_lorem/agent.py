from pathlib import Path

from google.adk import Agent

from .model import WORD_DELAY_SECONDS, MockLoremLlm

LOREM_PATH = Path(__file__).with_name("lorem_ipsum.txt")


def get_lorem_ipsum() -> dict:
    with open(LOREM_PATH, encoding="utf-8") as f:
        text = f.read().strip()
    return {"text": text}


def build_agent(*, word_delay_seconds: float = WORD_DELAY_SECONDS) -> Agent:
    return Agent(
        name="lorem_agent",
        model=MockLoremLlm(word_delay_seconds=word_delay_seconds),
        instruction="Always answer with lorem ipsum text from get_lorem_ipsum.",
        tools=[get_lorem_ipsum],
    )
