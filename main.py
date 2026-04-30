"""CLI-точка входа для генерации речи.

Пример:
    python main.py --topic "Будущее AI-агентов" --duration "7 минут"
    python main.py --topic "..." --duration "5 минут" --bio "..."  --output speech.md
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from config import DEFAULT_DURATION, DEFAULT_SPEAKER_BIO, DEFAULT_TOPIC, build_inputs
from graph import build_graph

_GRAPH = None


def get_graph():
    """Лениво строит граф один раз и кэширует — повторные вызовы переиспользуют его."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def generate_speech(
    topic: str,
    duration: str,
    speaker_bio: str = "",
    verbose: bool = True,
) -> str:
    """Запускает граф и возвращает финальный текст речи."""
    graph = get_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    inputs = build_inputs(topic=topic, duration=duration, speaker_bio=speaker_bio)

    if verbose:
        print(f"Тема: {topic}")
        print(f"Длительность: {duration}")
        if speaker_bio:
            print(f"Биография: {speaker_bio[:120]}{'...' if len(speaker_bio) > 120 else ''}")
        print("Запуск генерации речи...\n")

    start = time.time()
    for output in graph.stream(inputs, config=config, stream_mode="updates"):
        current_agent = next(iter(output))
        if verbose:
            print(f"Отработал: {current_agent} | {int(time.time() - start)}с")

    final_state = graph.get_state(config=config).values
    speech = final_state.get("result_speech", "") or ""
    speech = speech.replace("```markdown", "").replace("```", "").strip()
    return speech


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Мультиагентный спичрайтер на LangGraph + GigaChat")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Тема доклада")
    parser.add_argument("--duration", default=DEFAULT_DURATION, help="Длительность выступления (например, '5 минут')")
    parser.add_argument("--bio", default=DEFAULT_SPEAKER_BIO, help="Биография спикера (опционально)")
    parser.add_argument(
        "--output",
        default="result_speech.md",
        help="Файл для сохранения готовой речи (Markdown). По умолчанию result_speech.md",
    )
    parser.add_argument("--quiet", action="store_true", help="Не печатать промежуточные сообщения")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speech = generate_speech(
        topic=args.topic,
        duration=args.duration,
        speaker_bio=args.bio,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ РЕЧЬ:")
    print("=" * 60 + "\n")
    print(speech)

    Path(args.output).write_text(speech, encoding="utf-8")
    print(f"\nРечь сохранена в {args.output}")


if __name__ == "__main__":
    main()
