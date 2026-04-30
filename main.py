"""CLI-точка входа для ReAct-спичрайтера.

Пример:
    python main.py --topic "Будущее AI-агентов" --duration "7 минут"
    python main.py --topic "..." --duration "5 минут" --bio "..." --output speech.md
"""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

from config import DEFAULT_DURATION, DEFAULT_SPEAKER_BIO, DEFAULT_TOPIC
from graph import build_agent, last_speech, reset_state

_AGENT = None
DEFAULT_RECURSION_LIMIT = 40


def get_agent():
    """Лениво строит ReAct-агента один раз и кэширует."""
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


def _user_request(topic: str, duration: str, speaker_bio: str = "") -> str:
    parts = [f"Тема доклада: {topic}", f"Длительность: {duration}"]
    if speaker_bio:
        parts.append(f"Биография спикера: {speaker_bio}")
    parts.append("Сгенерируй и верни готовый текст речи в Markdown.")
    return "\n".join(parts)


def generate_speech(
    topic: str,
    duration: str,
    speaker_bio: str = "",
    verbose: bool = True,
    recursion_limit: int = DEFAULT_RECURSION_LIMIT,
) -> str:
    """Запускает ReAct-агента и возвращает финальный текст речи."""
    agent = get_agent()
    reset_state()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": recursion_limit,
    }
    inputs = {"messages": [("user", _user_request(topic, duration, speaker_bio))]}

    if verbose:
        print(f"Тема: {topic}")
        print(f"Длительность: {duration}")
        if speaker_bio:
            print(f"Биография: {speaker_bio[:120]}{'...' if len(speaker_bio) > 120 else ''}")
        print("Запуск ReAct-агента...\n")

    start = time.time()
    final_message = None

    for chunk in agent.stream(inputs, config=config, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            if not isinstance(node_state, dict):
                continue
            messages = node_state.get("messages") or []
            if not messages:
                continue
            last = messages[-1]
            final_message = last
            if not verbose:
                continue
            elapsed = int(time.time() - start)
            kind = type(last).__name__
            tool_calls = getattr(last, "tool_calls", None) or []
            if tool_calls:
                names = [tc.get("name") for tc in tool_calls]
                print(f"[{elapsed}s] {node_name}: {kind} → tool_calls: {names}")
            else:
                content = getattr(last, "content", "") or ""
                snippet = content[:80].replace("\n", " ")
                print(f"[{elapsed}s] {node_name}: {kind} → {snippet}")

    # Берём чистую (прошедшую self-verify в write_speech) последнюю речь из state. Если её нет —
    # как fallback используем content финального AIMessage.
    speech = last_speech()
    if not speech and final_message is not None:
        speech = getattr(final_message, "content", "") or ""
    speech = speech.replace("```markdown", "").replace("```", "").strip()
    return speech


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ReAct-спичрайтер на LangGraph + GigaChat")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Тема доклада")
    parser.add_argument("--duration", default=DEFAULT_DURATION, help="Длительность выступления")
    parser.add_argument("--bio", default=DEFAULT_SPEAKER_BIO, help="Биография спикера (опционально)")
    parser.add_argument(
        "--output",
        default="result_speech.md",
        help="Файл для сохранения готовой речи (Markdown). По умолчанию result_speech.md",
    )
    parser.add_argument("--quiet", action="store_true", help="Не печатать промежуточные сообщения")
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=DEFAULT_RECURSION_LIMIT,
        help="Лимит шагов LangGraph (по умолчанию 40)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speech = generate_speech(
        topic=args.topic,
        duration=args.duration,
        speaker_bio=args.bio,
        verbose=not args.quiet,
        recursion_limit=args.recursion_limit,
    )

    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ РЕЧЬ:")
    print("=" * 60 + "\n")
    print(speech)

    Path(args.output).write_text(speech, encoding="utf-8")
    print(f"\nРечь сохранена в {args.output}")


if __name__ == "__main__":
    main()
