"""Сборка ReAct-агента спичрайтера.

Используется ``langchain.agents.create_agent`` (LangChain 1.x; ранее это было
``langgraph.prebuilt.create_react_agent``) — стандартный ReAct-цикл
``(model → tool_calls → tool_results → model → ...)`` поверх 4 инструментов из tools.py.

Состояние агента — стандартный ``MessagesState``: история сообщений + tool calls + tool results.
Цикл завершается, когда модель перестаёт выдавать tool_calls и возвращает финальный ответ.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

import tools as tools_mod
from llm import giga
from loader import load_documents
from vectorstore import build_vectorstore

SYSTEM_PROMPT = """Ты - агент-спичрайтер. По теме доклада и длительности выступления ты должен подготовить
готовый текст речи в Markdown. Действуй по схеме ReAct: думай, вызывай инструменты, наблюдай результат, повторяй.

В твоём распоряжении 4 инструмента:
1. retrieve_quotes(query) - искать релевантные цитаты в векторной БД корпуса.
2. plan_speech(topic, duration, retrieved_quotes, speaker_bio) - составить план и ТЗ.
3. write_speech(topic, duration, plan, retrieved_quotes, speaker_bio, previous_version, critique) - написать
   или переписать речь.
4. critique_speech(speech, topic, duration, retrieved_quotes) - оценить речь, вернёт APPROVED или критику.

Алгоритм:
1. Вызови retrieve_quotes по теме доклада.
2. Вызови plan_speech, передав найденные цитаты.
3. Вызови write_speech по плану и цитатам.
4. Вызови critique_speech.
5. Если ответ критика - APPROVED, ВЕРНИ финальный ответ: ПОЛНЫЙ текст речи в Markdown без слова APPROVED
   и без своих комментариев.
6. Если критик дал замечания - вызови write_speech ещё раз, передав critique=<текст критики>
   и previous_version=<предыдущий вариант>. Если критик настаивает на нехватке цитат - сначала вызови
   retrieve_quotes с уточнённым запросом.
7. Не более 6 итераций цикла write→critique.

ВАЖНО:
- Финальный ответ - ТОЛЬКО текст речи в Markdown.
- Не пиши вступительных фраз вроде "Вот ваша речь:".
- Не выводи слово APPROVED в финальном ответе."""


def build_agent(checkpointer=None):
    """Загружает документы, поднимает Chroma, инжектит её в tools и собирает ReAct-агента."""
    print("Загрузка документов...")
    all_content = load_documents()

    print("Инициализация векторного хранилища...")
    vectorstore = build_vectorstore(all_content)
    tools_mod.set_vectorstore(vectorstore)

    agent = create_agent(
        giga,
        tools=tools_mod.ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or MemorySaver(),
    )
    print("ReAct-агент собран\n")
    return agent
