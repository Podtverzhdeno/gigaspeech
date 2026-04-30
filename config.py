"""Дефолтные значения для CLI и MCP. Используются, если пользователь не передал свои."""

DEFAULT_TOPIC = (
    "Влияние искусственного интеллекта и AI-агентов на современный бизнес"
)
DEFAULT_DURATION = "5 минут"
DEFAULT_SPEAKER_BIO = ""


def build_inputs(
    topic: str = DEFAULT_TOPIC,
    duration: str = DEFAULT_DURATION,
    speaker_bio: str = DEFAULT_SPEAKER_BIO,
) -> dict:
    """Собирает входное состояние графа из пользовательских значений."""
    return {
        "speech_topic": topic,
        "time_to_speak": duration,
        "speaker_bio": speaker_bio,
        "messages": [],
    }
