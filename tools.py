"""LangChain-инструменты для ReAct-агента спичрайтера.

Все 4 инструмента видны как `@tool`-функции и регистрируются в `langgraph.prebuilt.create_react_agent`.

- ``retrieve_quotes`` — поиск по векторной БД (детерминированный, без LLM).
- ``plan_speech``    — внутренний LLM-вызов: структура + ТЗ.
- ``write_speech``   — внутренний LLM-вызов: Markdown-текст речи.
- ``critique_speech`` — внутренний LLM-вызов: ``APPROVED`` или критика.

Векторная БД инжектится через ``set_vectorstore(vs)`` один раз при сборке агента.
"""
from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from llm import giga

_vectorstore: Any | None = None


def set_vectorstore(vs: Any) -> None:
    """Сохраняет ссылку на векторное хранилище для retrieve_quotes."""
    global _vectorstore
    _vectorstore = vs


# ---------------------------- retrieve_quotes ----------------------------


@tool
def retrieve_quotes(query: str) -> str:
    """Поиск релевантных цитат в корпусе документов через векторную БД.

    Используй этот инструмент:
    - В самом начале — чтобы собрать материалы по теме доклада.
    - Между итерациями переписывания, если критик попросил больше цитат.

    Args:
        query: Поисковый запрос (тема, тезис, ключевые слова).

    Returns:
        Текст с релевантными отрывками и указанием источника. Пустая строка, если ничего не найдено.
    """
    if _vectorstore is None:
        return "Векторная база не инициализирована."
    docs = _vectorstore.similarity_search(query[:2048], k=4)
    if not docs:
        return ""
    parts: list[str] = []
    for d in docs:
        source = d.metadata.get("document") or d.metadata.get("source") or "Источник"
        parts.append(f"(Цитата из «{source}»):\n{d.page_content}")
    return "\n\n".join(parts)


# ---------------------------- plan_speech --------------------------------


_PLAN_TEMPLATE = """Ты - помощник спичрайтера. Составь структуру речи и техническое задание.

Тема: {topic}
Длительность выступления: {duration}

Биография спикера: {speaker_bio}

Релевантные материалы из корпуса (могут быть пустыми):
{retrieved_quotes}

Верни ровно две секции:

СТРУКТУРА:
<разделы речи с примерным таймингом каждого>

ТЗ:
<инструкции спичрайтеру: тон, ключевые тезисы, требования к цитированию материалов>"""


@tool
def plan_speech(
    topic: str,
    duration: str,
    retrieved_quotes: str = "",
    speaker_bio: str = "",
) -> str:
    """Составить план речи и техническое задание спичрайтеру.

    Args:
        topic: Тема доклада.
        duration: Длительность выступления (например, "5 минут").
        retrieved_quotes: Цитаты, ранее полученные через retrieve_quotes (можно оставить пустым).
        speaker_bio: Биография спикера (опционально).

    Returns:
        Текст с двумя секциями: ``СТРУКТУРА:`` и ``ТЗ:``.
    """
    prompt = ChatPromptTemplate.from_messages([("system", _PLAN_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    return chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "speaker_bio": speaker_bio or "(не указана)",
            "retrieved_quotes": retrieved_quotes or "(пусто)",
        }
    )


# ---------------------------- write_speech -------------------------------


_WRITE_TEMPLATE = """Ты - спичрайтер. Напиши речь в Markdown по плану и материалам.

Тема: {topic}
Длительность: {duration}
Биография спикера: {speaker_bio}

План:
{plan}

Релевантные цитаты (используй 1-3, явно указывай источник в формате «(источник: <название>)»):
{retrieved_quotes}

Предыдущая версия речи (если есть, перепиши с учётом критики):
{previous_version}

Критика предыдущей версии:
{critique}

Требования:
- Markdown с заголовками; под каждым - примерное время чтения раздела.
- Если есть цитаты в материалах - вплети 1-3 с указанием источника.
- Не выдумывай цитаты, которых нет в материалах.
- Длина речи примерно соответствует указанной длительности.
- В ответе верни ТОЛЬКО текст речи, без мета-комментариев."""


@tool
def write_speech(
    topic: str,
    duration: str,
    plan: str,
    retrieved_quotes: str = "",
    speaker_bio: str = "",
    previous_version: str = "",
    critique: str = "",
) -> str:
    """Написать или переписать речь в формате Markdown.

    Args:
        topic: Тема доклада.
        duration: Длительность выступления.
        plan: План и ТЗ (например, результат plan_speech).
        retrieved_quotes: Цитаты для использования (из retrieve_quotes).
        speaker_bio: Биография спикера (опционально).
        previous_version: Предыдущая версия речи, если переписываем.
        critique: Критика предыдущей версии для переписывания.

    Returns:
        Готовый текст речи в Markdown.
    """
    prompt = ChatPromptTemplate.from_messages([("system", _WRITE_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    return chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "plan": plan,
            "retrieved_quotes": retrieved_quotes or "(пусто)",
            "speaker_bio": speaker_bio or "(не указана)",
            "previous_version": previous_version or "(нет)",
            "critique": critique or "(нет)",
        }
    )


# ---------------------------- critique_speech ----------------------------


_CRITIQUE_TEMPLATE = """Ты - толерантный выпускающий редактор. Оцени речь и реши, годится ли она к выступлению.

Тема: {topic}
Длительность: {duration}

Текст речи:
{speech}

Доступные материалы из корпуса:
{retrieved_quotes}

Правила (важны только КРИТИЧЕСКИЕ дефекты):
1. Тема раскрыта хотя бы на базовом уровне.
2. Есть хотя бы одна цитата с явно указанным источником из доступных материалов.
3. Длина соответствует таймингу (приблизительно, без перфекционизма).
4. Нет выдуманных цитат, которых нет в материалах.

Формат ответа СТРОГО один из двух вариантов:

ВАРИАНТ A. Если все 4 пункта выполнены - верни РОВНО одну строку без знаков препинания и без
кавычек, и больше ничего:
APPROVED

ВАРИАНТ B. Если нарушен хотя бы один пункт - верни нумерованный список 1-3 конкретных правок,
которые ОБЯЗАТЕЛЬНО надо внести. Не используй слова "НЕДОСТАТОЧНО", "НЕПОЛНО", "ПЕРЕСМОТРЕТЬ" и
прочие оценочные ярлыки - сразу пиши, что именно поправить.

Будь снисходителен: если речь "в целом нормальная", выбирай APPROVED. Не требуй идеала."""


@tool
def critique_speech(
    speech: str,
    topic: str,
    duration: str,
    retrieved_quotes: str = "",
) -> str:
    """Оценить речь и получить либо ``APPROVED``, либо конструктивную критику.

    Args:
        speech: Текущая версия речи.
        topic: Тема доклада.
        duration: Длительность выступления.
        retrieved_quotes: Доступные материалы из корпуса.

    Returns:
        Строка ``APPROVED`` (если речь готова) или текст критики.
    """
    prompt = ChatPromptTemplate.from_messages([("system", _CRITIQUE_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    return chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "speech": speech,
            "retrieved_quotes": retrieved_quotes or "(пусто)",
        }
    ).strip()


ALL_TOOLS = [retrieve_quotes, plan_speech, write_speech, critique_speech]
