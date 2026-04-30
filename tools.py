"""LangChain-инструменты для ReAct-агента спичрайтера.

Все 5 инструментов регистрируются как `@tool`-функции в `langchain.agents.create_agent`.

- ``retrieve_quotes``  — поиск по векторной БД (детерминированный, без LLM).
- ``plan_speech``      — внутренний LLM-вызов: структура + ТЗ.
- ``write_speech``     — внутренний LLM-вызов: Markdown-текст речи. Сам валидирует цитаты и
                         делает до 2 внутренних попыток исправления, если в речи появились
                         выдуманные источники.
- ``verify_quotes``    — детерминированная проверка: все ли цитаты в речи реально есть в корпусе.
- ``critique_speech``  — внутренний LLM-вызов: ``APPROVED`` или критика.

Вместо прокидывания длинных строк через аргументы tool_call (некоторые LLM, например GigaChat,
в этом случае подставляют плейсхолдер вроде «писавший текст выше» вместо реального текста),
длинные артефакты — `retrieved_quotes` и `last_speech` — хранятся в модульном state-словаре
``_STATE``. `retrieve_quotes` пишет туда найденные цитаты, `write_speech` пишет туда последнюю
версию речи; `verify_quotes`, `critique_speech` и повторный `write_speech` читают оттуда.

Перед каждым новым запуском агента вызывай ``reset_state()`` — иначе соседние сессии будут
видеть мусор от предыдущих.

Векторная БД инжектится через ``set_vectorstore(vs)`` один раз при сборке агента.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from llm import giga

_vectorstore: Any | None = None
_STATE: dict[str, str] = {
    "retrieved_quotes": "",
    "last_plan": "",
    "last_speech": "",
}


def set_vectorstore(vs: Any) -> None:
    """Сохраняет ссылку на векторное хранилище для retrieve_quotes / verify_quotes."""
    global _vectorstore
    _vectorstore = vs


def reset_state() -> None:
    """Очищает накопленный за прошлую сессию контекст. Вызывать в начале каждого запуска агента."""
    _STATE["retrieved_quotes"] = ""
    _STATE["last_plan"] = ""
    _STATE["last_speech"] = ""


# ============================ helpers ====================================


_QUOTE_PATTERNS = (
    r"«([^»]{15,})»",
    r"\"([^\"]{15,})\"",
    r"“([^”]{15,})”",
)
_MIN_MATCH_LEN = 30  # сколько подряд идущих символов цитаты должны быть найдены в корпусе


def _normalize(s: str) -> str:
    """Нормализация для нечёткого сравнения: lowercase + только буквы/цифры/пробел + один пробел."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^а-яёa-z0-9 ]+", "", s)
    return s.strip()


def _extract_quotes(text: str) -> list[str]:
    quotes: list[str] = []
    for pattern in _QUOTE_PATTERNS:
        quotes.extend(re.findall(pattern, text))
    seen: set[str] = set()
    out: list[str] = []
    for q in quotes:
        clean = q.strip()
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _quote_is_in_corpus(quote: str, corpus_norm: str) -> bool:
    norm = _normalize(quote)
    if len(norm) < _MIN_MATCH_LEN:
        return True
    needle = norm[:_MIN_MATCH_LEN]
    if needle in corpus_norm:
        return True
    if _vectorstore is None:
        return False
    docs = _vectorstore.similarity_search(quote[:512], k=4)
    for d in docs:
        if needle in _normalize(d.page_content):
            return True
    return False


def _find_fake_quotes(speech: str, retrieved_quotes: str) -> list[str]:
    quotes = _extract_quotes(speech)
    if not quotes:
        return []
    corpus_norm = _normalize(retrieved_quotes)
    return [q for q in quotes if not _quote_is_in_corpus(q, corpus_norm)]


def _strip_fake_quotes(speech: str, fakes: list[str]) -> str:
    """Удаляет строки с выдуманными цитатами как fallback на случай, если LLM не смог их выкинуть сам."""
    if not fakes:
        return speech
    out_lines: list[str] = []
    for line in speech.splitlines():
        if any(f in line for f in fakes):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


# ---------------------------- retrieve_quotes ----------------------------


@tool
def retrieve_quotes(query: str) -> str:
    """Поиск релевантных цитат в корпусе документов через векторную БД.

    Используй этот инструмент:
    - В самом начале — чтобы собрать материалы по теме доклада.
    - Между итерациями переписывания, если в речи не хватает цитат.

    Результат автоматически кэшируется в модульном state — write_speech / verify_quotes /
    critique_speech читают его оттуда, передавать его повторно как аргумент НЕ нужно.

    Args:
        query: Поисковый запрос (тема, тезис, ключевые слова).

    Returns:
        Текст с релевантными отрывками и указанием источника. Пустая строка, если ничего не найдено.
    """
    if _vectorstore is None:
        return "Векторная база не инициализирована."
    docs = _vectorstore.similarity_search(query[:2048], k=4)
    if not docs:
        _STATE["retrieved_quotes"] = ""
        return ""
    parts: list[str] = []
    for d in docs:
        source = d.metadata.get("document") or d.metadata.get("source") or "Источник"
        parts.append(f"(Цитата из «{source}»):\n{d.page_content}")
    text = "\n\n".join(parts)
    _STATE["retrieved_quotes"] = text
    return text


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
    speaker_bio: str = "",
) -> str:
    """Составить план речи и техническое задание спичрайтеру.

    Использует материалы из последнего вызова retrieve_quotes автоматически — передавать их
    как аргумент не нужно.

    Args:
        topic: Тема доклада.
        duration: Длительность выступления (например, "5 минут").
        speaker_bio: Биография спикера (опционально).

    Returns:
        Текст с двумя секциями: ``СТРУКТУРА:`` и ``ТЗ:``.
    """
    prompt = ChatPromptTemplate.from_messages([("system", _PLAN_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    text = chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "speaker_bio": speaker_bio or "(не указана)",
            "retrieved_quotes": _STATE["retrieved_quotes"] or "(пусто)",
        }
    )
    _STATE["last_plan"] = text
    return text


# ---------------------------- write_speech -------------------------------


_WRITE_TEMPLATE = """Ты - спичрайтер. Напиши речь в Markdown по плану и материалам.

Тема: {topic}
Длительность: {duration}
Биография спикера: {speaker_bio}

План:
{plan}

Релевантные цитаты из корпуса (это ЕДИНСТВЕННЫЙ источник цитат для этой речи):
{retrieved_quotes}

Предыдущая версия речи (если есть, перепиши с учётом критики):
{previous_version}

Критика предыдущей версии:
{critique}

Требования:
- Markdown с заголовками; под каждым - примерное время чтения раздела.
- Длина речи примерно соответствует указанной длительности.
- В ответе верни ТОЛЬКО текст речи, без мета-комментариев.

ЖЁСТКИЕ ПРАВИЛА ПО ЦИТАТАМ (нарушение = провал валидации):
- Цитировать в кавычках «...» можно ИСКЛЮЧИТЕЛЬНО фрагменты из блока «Релевантные цитаты из корпуса»
  выше. Каждая цитата должна быть взята оттуда дословно.
- ЗАПРЕЩЕНО ссылаться на «McKinsey», «Сбербанк», «WEF», «IDC», «Gartner», «Основы AI» и любые
  другие источники, которых нет в блоке «Релевантные цитаты из корпуса». Не выдумывай отчёты,
  статистики и пресс-релизы.
- Если в блоке нет подходящей цитаты для пункта - просто напиши пункт БЕЗ цитаты.
- Каждую использованную цитату подписывай в формате «(источник: <название из блока>)»."""


_MAX_WRITE_RETRIES = 2


def _llm_write(
    topic: str,
    duration: str,
    plan: str,
    retrieved_quotes: str,
    speaker_bio: str,
    previous_version: str,
    critique: str,
) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", _WRITE_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    return chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "plan": plan or "(нет плана)",
            "retrieved_quotes": retrieved_quotes or "(пусто)",
            "speaker_bio": speaker_bio or "(не указана)",
            "previous_version": previous_version or "(нет)",
            "critique": critique or "(нет)",
        }
    )


@tool
def write_speech(
    topic: str,
    duration: str,
    speaker_bio: str = "",
    critique: str = "",
) -> str:
    """Написать или переписать речь в формате Markdown.

    Берёт план из последнего plan_speech, цитаты из последнего retrieve_quotes и предыдущую
    версию из последнего write_speech (всё через модульный state). Если в готовой речи
    обнаруживаются выдуманные цитаты — внутренне переписывает речь до 2 раз с явным указанием
    «удали такие-то фейковые цитаты»; если LLM упорствует, удаляет строки с фейковыми цитатами
    программно.

    Результат сохраняется в state и доступен verify_quotes / critique_speech без передачи.

    Args:
        topic: Тема доклада.
        duration: Длительность выступления.
        speaker_bio: Биография спикера (опционально).
        critique: Критика предыдущей версии речи (для второй и далее итераций).

    Returns:
        Готовый текст речи в Markdown без выдуманных цитат.
    """
    plan = _STATE["last_plan"]
    retrieved = _STATE["retrieved_quotes"]
    previous = _STATE["last_speech"]

    text = _llm_write(topic, duration, plan, retrieved, speaker_bio, previous, critique)

    for attempt in range(_MAX_WRITE_RETRIES):
        fakes = _find_fake_quotes(text, retrieved)
        if not fakes:
            break
        fake_list = "\n".join(f"- «{q[:200]}»" for q in fakes)
        retry_critique = (
            (critique + "\n\n" if critique else "")
            + "В предыдущей версии есть выдуманные цитаты, которых НЕТ в материалах корпуса. "
            + "Удали их полностью или замени на дословные фрагменты из блока 'Релевантные цитаты "
            + "из корпуса'. Список выдуманных цитат:\n"
            + fake_list
        )
        text = _llm_write(topic, duration, plan, retrieved, speaker_bio, text, retry_critique)

    # Финальный жёсткий fallback: если LLM не справилась за 2 попытки, выпиливаем фейки регуляркой.
    fakes = _find_fake_quotes(text, retrieved)
    if fakes:
        text = _strip_fake_quotes(text, fakes)

    _STATE["last_speech"] = text
    return text


# ---------------------------- verify_quotes ------------------------------


@tool
def verify_quotes() -> str:
    """Проверить, что каждая цитата в последней версии речи реально присутствует в корпусе.

    Не принимает аргументов — берёт текст последней речи и список retrieved_quotes из state.

    Returns:
        Строка ``OK`` если все цитаты подтверждены или цитат нет.
        Иначе многострочный список выдуманных цитат, начинающийся со слова ``FAKE``.
    """
    speech = _STATE["last_speech"]
    if not speech:
        return "OK (речи ещё нет — сначала вызови write_speech)"
    retrieved = _STATE["retrieved_quotes"]
    fakes = _find_fake_quotes(speech, retrieved)
    if not fakes:
        return "OK"
    lines = ["FAKE (этих цитат нет в корпусе):"]
    for q in fakes:
        snippet = q if len(q) <= 200 else q[:200] + "..."
        lines.append(f"- «{snippet}»")
    return "\n".join(lines)


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
2. Длина соответствует таймингу (приблизительно, без перфекционизма).
3. Нет выдуманных цитат, которых нет в материалах (если цитат нет вообще — это допустимо).
4. Структура осмысленная.

Формат ответа СТРОГО один из двух вариантов:

ВАРИАНТ A. Если все 4 пункта выполнены - верни РОВНО одну строку без знаков препинания и без
кавычек, и больше ничего:
APPROVED

ВАРИАНТ B. Если нарушен хотя бы один пункт - верни нумерованный список 1-3 конкретных правок,
которые ОБЯЗАТЕЛЬНО надо внести. Не используй слова "НЕДОСТАТОЧНО", "НЕПОЛНО", "ПЕРЕСМОТРЕТЬ" и
прочие оценочные ярлыки - сразу пиши, что именно поправить.

Будь снисходителен: если речь "в целом нормальная", выбирай APPROVED. Не требуй идеала."""


@tool
def critique_speech(topic: str, duration: str) -> str:
    """Оценить последнюю версию речи и получить либо ``APPROVED``, либо конструктивную критику.

    Не принимает текст речи и цитаты как аргументы — берёт их из модульного state.

    Args:
        topic: Тема доклада (для контекста).
        duration: Длительность выступления (для контекста).

    Returns:
        Строка ``APPROVED`` (если речь годится) или текст критики.
    """
    speech = _STATE["last_speech"]
    if not speech:
        return "Нет речи для критики — сначала вызови write_speech."
    retrieved = _STATE["retrieved_quotes"]
    prompt = ChatPromptTemplate.from_messages([("system", _CRITIQUE_TEMPLATE)])
    chain = prompt | giga | StrOutputParser()
    return chain.invoke(
        {
            "topic": topic,
            "duration": duration,
            "speech": speech,
            "retrieved_quotes": retrieved or "(пусто)",
        }
    ).strip()


ALL_TOOLS = [retrieve_quotes, plan_speech, write_speech, verify_quotes, critique_speech]
