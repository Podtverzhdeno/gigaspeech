"""MCP-сервер для спичрайтера.

Exposes a single MCP tool, ``generate_speech(topic, duration, speaker_bio="")``, which runs the
multi-agent LangGraph pipeline and returns the resulting speech as Markdown text. Uses the official
MCP Python SDK (`mcp[cli]`).

Запуск напрямую (stdio):
    python mcp_server.py

Регистрация в Claude Desktop / Cursor / любом другом MCP-клиенте — см. README.md.
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

    Args:
        topic: Тема доклада. Например: "Будущее AI-агентов в Web3".
        duration: Целевое время выступления. Например: "5 минут", "10 минут".
        speaker_bio: Опциональная биография спикера, чтобы агенты адаптировали тон.

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
    """Возвращает текущую конфигурацию по умолчанию (для дебага из MCP-клиента)."""
    return {
        "default_topic": DEFAULT_TOPIC,
        "default_duration": DEFAULT_DURATION,
        "giga_credentials_set": bool(os.getenv("GIGA_CREDENTIALS")),
    }


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
