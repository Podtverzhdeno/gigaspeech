"""MCP-сервер для ReAct-спичрайтера.

Поднимает MCP-сервер по протоколу stdio и публикует один инструмент:
``generate_speech_tool(topic, duration, speaker_bio="")``. Внутри запускается тот же
ReAct-агент из ``graph.py``, что и в CLI.

Запуск:
    python mcp_server.py
    speech-writer-mcp           # если установлен пакет

Регистрация в Claude Desktop / Cursor — см. README.md.
"""
from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

from config import DEFAULT_DURATION, DEFAULT_TOPIC
from main import generate_speech

mcp = FastMCP("speech-writer")


@mcp.tool()
def generate_speech_tool(
    topic: str = DEFAULT_TOPIC,
    duration: str = DEFAULT_DURATION,
    speaker_bio: str = "",
) -> str:
    """Сгенерировать текст публичного выступления по теме и длительности.

    Запускает ReAct-агента, который сам решает, какие инструменты вызывать
    (поиск цитат, планирование, написание, критика), и возвращает финальный
    текст речи в Markdown.

    Args:
        topic: Тема доклада. Например: "Будущее AI-агентов в Web3".
        duration: Целевое время выступления. Например: "5 минут", "10 минут".
        speaker_bio: Опциональная биография спикера, чтобы агент адаптировал тон.

    Returns:
        Готовый текст речи в формате Markdown.
    """
    return generate_speech(
        topic=topic,
        duration=duration,
        speaker_bio=speaker_bio,
        verbose=False,
    )


@mcp.resource("speechwriter://config")
def get_config() -> dict[str, Any]:
    """Текущая конфигурация по умолчанию (для дебага из MCP-инспектора)."""
    return {
        "default_topic": DEFAULT_TOPIC,
        "default_duration": DEFAULT_DURATION,
        "giga_credentials_set": bool(os.getenv("GIGA_CREDENTIALS")),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
