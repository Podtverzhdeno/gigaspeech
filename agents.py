"""Агенты мультиагентной системы спичрайтера.

Архитектура (см. graph.py):
    START → Retriever → Planner → Writer → Critic ↔ {Writer, Retriever, END}

Все промпты обобщены: они работают с произвольной темой и любым корпусом документов,
полученным из векторной БД, без жёсткой привязки к конкретным книгам или нормативным актам.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt

from llm import giga
from state import SpeechWriterState

WRITER_NODE = "writer"
CRITIC_NODE = "critic"
PLANNER_NODE = "planner"
RETRIEVER_NODE = "retriever"


# ----------------------------- Planner -----------------------------------


class SpeechPlan(BaseModel):
    """Структурный результат работы Planner."""

    speech_structure: str = Field(description="Структура речи, релевантная теме доклада.")
    speech_tech_spec: str = Field(description="Техническое задание спичрайтеру.")
    relevant_quotes: list[str] = Field(
        description=(
            "Список релевантных теме цитат/тезисов, извлечённых из материалов корпуса. "
            "Не менее 5 пунктов; цитаты — дословные фрагменты из RETRIEVED_DOCS."
        )
    )


PLAN_PROMPT = """Ты - агент-аналитик и помощник спичрайтера. Твоя задача - на основе темы доклада и материалов из RAG-корпуса
составить структуру и техническое задание для написания речи.

Тема доклада: {speech_topic}
Время выступления: {time_to_speak}

Биография спикера (если предоставлена, учитывай тон и контекст; если пустая - игнорируй):
<BIO>
{speaker_bio}
</BIO>

Релевантные материалы из векторной базы знаний (используй как основной источник цитат и фактов):
<RETRIEVED_DOCS>
{retriever_docs}
</RETRIEVED_DOCS>

Требования:
- Структура должна логично укладываться в указанное время выступления.
- В relevant_quotes выбирай только дословные фрагменты, действительно присутствующие в RETRIEVED_DOCS.
- Если корпус пуст, верни пустой список relevant_quotes и всё равно предложи структуру.

Выведи только JSON в следующем формате:
{format_instructions}"""


@retry(stop=stop_after_attempt(3))
def plan_speech(state: SpeechWriterState) -> dict:
    print("Planner начинает работу")

    parser = PydanticOutputParser(pydantic_object=SpeechPlan)
    prompt = ChatPromptTemplate.from_messages([("system", PLAN_PROMPT)]).partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | giga | parser
    res = chain.invoke(
        {
            "speech_topic": state["speech_topic"],
            "time_to_speak": state["time_to_speak"],
            "speaker_bio": state.get("speaker_bio", "") or "(не указана)",
            "retriever_docs": state.get("retriever_docs", "") or "(пусто)",
        }
    )

    return {
        "speech_structure": res.speech_structure,
        "speech_tech_spec": res.speech_tech_spec,
        "relevant_quotes": res.relevant_quotes,
    }


# ----------------------------- Writer ------------------------------------


WRITE_PROMPT = """Ты - агент-спичрайтер. Пишешь речь для публичного выступления на основании плана и материалов из корпуса.

Тема: {speech_topic}
Время выступления: {time_to_speak}

Биография спикера:
<BIO>
{speaker_bio}
</BIO>

Ключевые тезисы и цитаты, отобранные планировщиком:
<KEY_QUOTES>
{relevant_quotes}
</KEY_QUOTES>

План выступления:
<SPEECH_PLAN>
Структура: {speech_structure}
ТЗ: {speech_tech_spec}
</SPEECH_PLAN>

Материалы из корпуса (используй для цитирования; указывай название источника):
<RETRIEVED_DOCS>
{retriever_docs}
</RETRIEVED_DOCS>

Предыдущая версия речи (если есть - перепиши с учётом критики):
<OLD_SPEECH>
{result_speech}
</OLD_SPEECH>

Критика предыдущей версии:
<CRITIQUE>
{critique}
</CRITIQUE>

Требования:
- Если RETRIEVED_DOCS не пустой - вплети 1-3 цитаты, явно указав источник в формате: «(источник: <название>)».
- Не выдумывай цитаты, которых нет в RETRIEVED_DOCS.
- Длина речи должна примерно соответствовать указанному времени выступления.
- Оформи речь в Markdown: разделы с заголовками, под каждым разделом - примерное время чтения.
- Не вставляй мета-комментарии вроде «вот ваша речь»; выведи только сам текст речи."""


def write_speech(state: SpeechWriterState) -> dict:
    docs = state.get("retriever_docs", "") or ""
    critique_list = state.get("critique", []) or [""]
    critique = critique_list[-1] if critique_list else ""
    print(f"Writer начинает работу | объём цитат из корпуса: {len(docs)} символов")

    prompt = ChatPromptTemplate.from_messages([("system", WRITE_PROMPT)])
    chain = prompt | giga | StrOutputParser()

    result = chain.invoke(
        {
            "speech_topic": state["speech_topic"],
            "time_to_speak": state["time_to_speak"],
            "speaker_bio": state.get("speaker_bio", "") or "(не указана)",
            "relevant_quotes": state.get("relevant_quotes", []),
            "speech_structure": state.get("speech_structure", ""),
            "speech_tech_spec": state.get("speech_tech_spec", ""),
            "result_speech": state.get("result_speech", ""),
            "critique": critique,
            "retriever_docs": docs or "(пусто)",
        }
    )
    return {"result_speech": result}


# ----------------------------- Critic ------------------------------------


class CritiqueResult(BaseModel):
    """Структурный результат работы Critic."""

    thoughts: str = Field(description="Мысли по поводу написанной речи.")
    critique: str = Field(description="Конструктивная критика - что нужно поправить.")
    is_new_critique: bool = Field(
        description="Есть ли принципиально новая критика по сравнению с предыдущими итерациями."
    )
    final_decision: str = Field(
        description="Итоговое решение: good (готово), retrieve (нужно ещё цитат из корпуса), fix (переписать)."
    )


CRITIQUE_PROMPT = """Ты - агент-выпускающий редактор. Оцениваешь черновик речи и выбираешь следующий шаг.

Тема: {speech_topic}
Время выступления: {time_to_speak}

Биография спикера:
<BIO>
{speaker_bio}
</BIO>

Текущий текст речи:
<SPEECH>
{result_speech}
</SPEECH>

Доступные материалы из корпуса:
<RETRIEVED_DOCS>
{retriever_docs}
</RETRIEVED_DOCS>

Старая критика (не повторяйся):
<OLD_CRITIQUE>
{old_critique}
</OLD_CRITIQUE>

Правила принятия решения:
- Если RETRIEVED_DOCS пустой ИЛИ в речи нет ни одной явной цитаты с указанием источника → final_decision = retrieve.
- Если есть принципиально новая конструктивная критика → final_decision = fix (и is_new_critique = true).
- Если речь соответствует теме, тайминг адекватен и новой критики нет → final_decision = good.

Выведи только JSON:
{format_instructions}"""


MAX_CRITIQUE_ITERATIONS = 8


@retry(stop=stop_after_attempt(3))
def critique_speech(
    state: SpeechWriterState,
) -> Command[Literal["writer", "retriever", "__end__"]]:
    print("Critic начинает работу")

    parser = PydanticOutputParser(pydantic_object=CritiqueResult)
    prompt = ChatPromptTemplate.from_messages([("system", CRITIQUE_PROMPT)]).partial(
        format_instructions=parser.get_format_instructions()
    )

    chain = prompt | giga | parser
    resp = chain.invoke(
        {
            "speech_topic": state["speech_topic"],
            "time_to_speak": state["time_to_speak"],
            "speaker_bio": state.get("speaker_bio", "") or "(не указана)",
            "result_speech": state.get("result_speech", ""),
            "retriever_docs": state.get("retriever_docs", "") or "(пусто)",
            "old_critique": state.get("critique", []),
        }
    )

    decision = resp.final_decision
    old_critique = list(state.get("critique", []) or [])
    old_critique.append(resp.critique)

    print(f"  решение: {decision} | новая критика: {resp.is_new_critique}")

    if len(old_critique) >= MAX_CRITIQUE_ITERATIONS:
        print("Достигнут лимит итераций критики — завершаем")
        return Command(update={"critique": old_critique}, goto="__end__")

    if decision == "retrieve" or not state.get("retriever_docs"):
        goto = RETRIEVER_NODE
    elif decision == "fix" and resp.is_new_critique:
        goto = WRITER_NODE
    else:
        goto = "__end__"

    return Command(update={"critique": old_critique}, goto=goto)


# ----------------------------- Retriever ---------------------------------


def retrieve_docs(state: SpeechWriterState, vectorstore) -> dict:
    """Ищет релевантные отрывки из корпуса по теме (на первой итерации) или по тексту речи."""
    print("Retriever начинает работу")

    query = state.get("result_speech") or state["speech_topic"]
    query = query[:2048]
    docs = vectorstore.similarity_search(query, k=4)

    old_docs = state.get("retriever_docs", "") or ""
    new_docs = old_docs

    for doc in docs:
        if doc.page_content not in new_docs:
            source = doc.metadata.get("document") or doc.metadata.get("source") or "Источник"
            new_docs += f"(Цитата из «{source}»):\n{doc.page_content}\n\n"

    return {"retriever_docs": new_docs}
