"""Сборка LangGraph-графа спичрайтера.

Граф:
    START → Retriever → Planner → Writer → Critic
                ↑                              │
                └────── retrieve ──────────────┤
                                               ├─── fix ──→ Writer
                                               └─── good ──→ END

Retriever запускается первым, чтобы Planner видел релевантные материалы для произвольной темы.
"""
from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from agents import (
    CRITIC_NODE,
    PLANNER_NODE,
    RETRIEVER_NODE,
    WRITER_NODE,
    critique_speech,
    plan_speech,
    retrieve_docs,
    write_speech,
)
from loader import load_documents
from state import SpeechWriterState
from vectorstore import build_vectorstore


def build_graph():
    print("Загрузка документов...")
    all_content = load_documents()

    print("Инициализация векторного хранилища...")
    vectorstore = build_vectorstore(all_content)

    retriever = partial(retrieve_docs, vectorstore=vectorstore)

    builder = StateGraph(SpeechWriterState)
    builder.add_node(RETRIEVER_NODE, retriever)
    builder.add_node(PLANNER_NODE, plan_speech)
    builder.add_node(WRITER_NODE, write_speech)
    builder.add_node(CRITIC_NODE, critique_speech)

    builder.add_edge(START, RETRIEVER_NODE)
    builder.add_edge(RETRIEVER_NODE, PLANNER_NODE)
    builder.add_edge(PLANNER_NODE, WRITER_NODE)
    builder.add_edge(WRITER_NODE, CRITIC_NODE)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    print("Граф собран\n")
    return graph
