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

ВАЖНО: длинные тексты (найденные цитаты, план, текущая версия речи) хранятся внутри инструментов.
В аргументах инструментам передавай только короткие значения: тему, длительность, биографию,
короткую критику. Не нужно копировать огромные блоки текста в tool_call.

5 инструментов:
1. retrieve_quotes(query) - найти релевантные цитаты в векторной БД корпуса. Сохраняет результат
   во внутренний state, чтобы остальные инструменты его использовали.
2. plan_speech(topic, duration, speaker_bio="") - составить план и ТЗ; сам подтягивает найденные цитаты.
3. write_speech(topic, duration, speaker_bio="", critique="") - написать или переписать речь;
   сам подтягивает план, цитаты и предыдущую версию. ВНУТРИ САМ ВАЛИДИРУЕТ цитаты на отсутствие
   фейков и переписывает себя при необходимости. Аргумент `critique` нужен только для второй
   и далее итераций.
4. verify_quotes() - проверить последнюю версию речи на выдуманные цитаты. Возвращает OK или FAKE.
   В норме после write_speech всегда возвращает OK (write_speech уже сам себя валидирует), но
   на всякий случай можно дёрнуть.
5. critique_speech(topic, duration) - оценить последнюю версию речи; вернёт APPROVED или критику.

Алгоритм:
1. retrieve_quotes(query="<тема доклада или связанные ключевые слова>").
2. plan_speech(topic, duration, speaker_bio).
3. write_speech(topic, duration, speaker_bio) - итерация №1.
4. critique_speech(topic, duration).
5. Если ответ критика - именно одно слово APPROVED, ВЕРНИ финальный ответ: ПОЛНЫЙ текст речи в
   Markdown БЕЗ слова APPROVED и БЕЗ своих комментариев. Текст бери из последнего ответа write_speech.
6. Иначе - write_speech(topic, duration, speaker_bio, critique="<текст критики>") - итерация №2,
   снова critique_speech.
7. Если после 3 итераций write→critique критик всё ещё не APPROVED - возвращай последнюю версию
   речи как финальный ответ.

ОЧЕНЬ ВАЖНЫЕ ОГРАНИЧЕНИЯ:
- НИКОГДА не вызывай critique_speech два раза подряд без write_speech между ними.
- ВСЕГО НЕ БОЛЕЕ 3 вызовов write_speech за сессию.
- Финальный ответ - ТОЛЬКО текст речи в Markdown.
- Не пиши вступительных фраз вроде "Вот ваша речь:".
- Не выводи слово APPROVED в финальном ответе.
- Если что-то идёт не так - всё равно верни последнюю успешную версию речи как финальный ответ."""


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


def reset_state() -> None:
    """Сбросить накопленный state инструментов перед новым запуском агента."""
    tools_mod.reset_state()


def last_speech() -> str:
    """Текущее значение state['last_speech'] — гарантированно прошло self-verify."""
    return tools_mod._STATE.get("last_speech", "")
