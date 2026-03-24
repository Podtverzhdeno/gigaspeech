from typing import Literal
from tenacity import retry, stop_after_attempt
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langgraph.types import Command

from state import SpeechWriterState
from llm import giga, chatgpt

class SpeechPlan(BaseModel):
    """Выходная схема планировщика."""
    speech_structure: str = Field(description="Структура речи, релевантной тематике мероприятия.")
    speech_tech_spec: str = Field(description="Техническое задание на написание речи.")
    relevant_quotes_309: list[str] = Field(
        description="Релевантные теме мероприятия цитаты из указа №309, не менее 10 предложений"
    )

PLAN_PROMPT = """Ты - агент-аналитик, который занимается помощью в составлении отличных речей для публичных выступлений.
Ты должен помочь спичрайтеру написать хорошую речь, поэтому ты должен составить техническое задание на написание речи.
Тема выступления - {speech_topic}. Время выступления - {time_to_speak}.

Биография спикера:
<BIO>
{speaker_bio}
</BIO>

Основной документ — Указ №309 "О национальных целях развития РФ до 2036 года":
<MAIN_DOCUMENT>
{content_309}
</MAIN_DOCUMENT>

Информация о мероприятии:
<EVENT_INFO>
{event_info}
</EVENT_INFO>

Выведи только следующую информацию в формате JSON:
{format_instructions}"""

@retry(stop=stop_after_attempt(3))
def plan_speech(state: SpeechWriterState, all_content: dict) -> dict:
    print(" Planner начинает работу")

    parser = PydanticOutputParser(pydantic_object=SpeechPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLAN_PROMPT)
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | giga | parser
    res = chain.invoke({
        "speech_topic": state["speech_topic"],
        "time_to_speak": state["time_to_speak"],
        "speaker_bio": state["speaker_bio"],
        "content_309": all_content["ukaz_309.md"],
        "event_info": all_content["ai_conf.md"],
    })

    return {
        "speech_structure": res.speech_structure,
        "speech_tech_spec": res.speech_tech_spec,
        "relevant_quotes_309": res.relevant_quotes_309,
    }

WRITE_PROMPT = """Ты - агент-спичрайтер. Пишешь речь для публичных личностей на основании данных документов.
Тема выступления - {speech_topic}. Время выступления - {time_to_speak}.

Биография спикера:
<BIO>
{speaker_bio}
</BIO>

Релевантные части указа №309 (ОБЯЗАТЕЛЬНО сослаться в речи):
<QUOTES_309>
{relevant_quotes_309}
</QUOTES_309>

Информация о мероприятии:
<EVENT_INFO>
{event_info}
</EVENT_INFO>

План выступления от коллеги-планировщика:
<SPEECH_PLAN>
Структура: {speech_structure}
ТЗ: {speech_tech_spec}
</SPEECH_PLAN>

Предыдущая версия речи (если есть — исправь с учётом критики):
<OLD_SPEECH>
{result_speech}
</OLD_SPEECH>

Критика предыдущей версии:
<CRITIQUE>
{critique}
</CRITIQUE>

Отрывки из книг для ОБЯЗАТЕЛЬНОГО цитирования (укажи явно название книги):
<BOOK_DOCS>
{retriever_docs}
</BOOK_DOCS>

Требования:
- Обязательно процитируй книги, если отрывки предоставлены. Укажи явно название книги.
- Обязательно сошлись на Указ №309.
- Не придумывай цитаты, которых нет в предложенных отрывках.
- Оформи речь в Markdown: разделы с заголовками, время чтения под каждым заголовком."""

def write_speech(state: SpeechWriterState, all_content: dict) -> dict:
    docs = state.get("retriever_docs", "")
    critique = state.get("critique", [""])[-1]
    print(f"Writer начинает работу | Объём цитат из книг: {len(docs)} символов")

    prompt = ChatPromptTemplate.from_messages([("system", WRITE_PROMPT)])
    chain = prompt | chatgpt | StrOutputParser()

    result = chain.invoke({
        "speech_topic": state["speech_topic"],
        "time_to_speak": state["time_to_speak"],
        "speaker_bio": state["speaker_bio"],
        "relevant_quotes_309": state["relevant_quotes_309"],
        "event_info": all_content["ai_conf.md"],
        "speech_structure": state["speech_structure"],
        "speech_tech_spec": state["speech_tech_spec"],
        "result_speech": state.get("result_speech", ""),
        "critique": critique,
        "retriever_docs": docs,
    })

    return {"result_speech": result}

class CritiqueResult(BaseModel):
    """Выходная схема критика."""
    thoughts: str = Field(description="Мысли по поводу написанной речи")
    critique: str = Field(description="Конструктивная критика — что нужно поправить")
    is_new_critique: bool = Field(description="Есть ли принципиально новая критика по сравнению со старой")
    final_decision: str = Field(
        description="Итоговое решение: good (речь готова), retrieve (нужны цитаты из книг), fix (нужно переписать)"
    )

CRITIQUE_PROMPT = """Ты - агент-выпускающий редактор. Оцениваешь речь и решаешь — готова ли она или нужна доработка.
Тема выступления - {speech_topic}. Время выступления - {time_to_speak}.

Биография спикера:
<BIO>
{speaker_bio}
</BIO>

Релевантные части указа №309 (должны быть упомянуты в речи):
<QUOTES_309>
{relevant_quotes_309}
</QUOTES_309>

Информация о мероприятии:
<EVENT_INFO>
{event_info}
</EVENT_INFO>

Текущий текст речи:
<SPEECH>
{result_speech}
</SPEECH>

Доступные отрывки из книг:
<BOOK_DOCS>
{retriever_docs}
</BOOK_DOCS>

Старая критика (не повторяйся):
<OLD_CRITIQUE>
{old_critique}
</OLD_CRITIQUE>

Правила принятия решения:
- Если BOOK_DOCS пустой → final_decision = retrieve
- Если речь не содержит явных цитат из книг → final_decision = retrieve
- Если есть новая конструктивная критика → final_decision = fix
- Если критики нет ничего нового → final_decision = good

Выведи только JSON:
{format_instructions}"""

@retry(stop=stop_after_attempt(3))
def critique_speech(state: SpeechWriterState, all_content: dict) -> Command[Literal["👨‍💻 Writer", "🌐 Retriever"]]:
    print(" Critic начинает работу")

    parser = PydanticOutputParser(pydantic_object=CritiqueResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIQUE_PROMPT)
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | giga | parser
    resp = chain.invoke({
        "speech_topic": state["speech_topic"],
        "time_to_speak": state["time_to_speak"],
        "speaker_bio": state["speaker_bio"],
        "relevant_quotes_309": state["relevant_quotes_309"],
        "event_info": all_content["ai_conf.md"],
        "result_speech": state.get("result_speech", ""),
        "retriever_docs": state.get("retriever_docs", ""),
        "old_critique": state.get("critique", []),
    })

    decision = resp.final_decision
    old_critique = state.get("critique", [])
    old_critique.append(resp.critique)

    print(f"   Решение: {decision} | Новая критика: {resp.is_new_critique}")

    if len(old_critique) >= 15:
        print(" Достигнут лимит итераций критики — завершаем")
        return Command(update={"critique": old_critique}, goto="__end__")

    if decision == "retrieve" or not state.get("retriever_docs"):
        goto = " Retriever"
    elif decision == "fix" and resp.is_new_critique:
        goto = " Writer"
    else:
        goto = "__end__"

    return Command(update={"critique": old_critique}, goto=goto)


def retrieve_docs(state: SpeechWriterState, vectorstore) -> dict:
    """Ищет релевантные отрывки из книг Маркова по тексту текущей речи."""
    print("Retriever начинает работу")

    query = state.get("result_speech", state["speech_topic"])[:2048]
    docs = vectorstore.similarity_search(query, k=4)

    old_docs = state.get("retriever_docs", "")
    new_docs = old_docs

    for doc in docs:
        if doc.page_content not in new_docs:
            book_title = doc.metadata.get("document", "Книга Маркова")
            new_docs += f"(Цитата из книги «{book_title}»):\n{doc.page_content}\n\n"

    return {"retriever_docs": new_docs}
