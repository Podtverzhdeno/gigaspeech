from functools import partial
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from state import SpeechWriterState
from agents import plan_speech, write_speech, critique_speech, retrieve_docs
from loader import load_documents
from vectorstore import build_vectorstore


def build_graph():
    """
    Загружает документы, строит векторное хранилище и компилирует граф.

    Граф:
        START → Planner → Writer → Critic
                              ↑         ↓
                         Retriever ← (retrieve)
                                        ↓
                                       END
    """
    print("Загрузка документов...")
    all_content = load_documents()

    print("\n Инициализация векторного хранилища...")
    vectorstore = build_vectorstore(all_content)

    planner    = partial(plan_speech,    all_content=all_content)
    writer     = partial(write_speech,   all_content=all_content)
    critic     = partial(critique_speech, all_content=all_content)
    retriever  = partial(retrieve_docs,  vectorstore=vectorstore)

    builder = StateGraph(SpeechWriterState)
    builder.add_node(" Planner",    planner)
    builder.add_node(" Writer",    writer)
    builder.add_node(" Critic",   critic)
    builder.add_node(" Retriever",  retriever)

    builder.add_edge(START,           " Planner")
    builder.add_edge(" Planner",    " Writer")
    builder.add_edge(" Writer",    " Critic")
    builder.add_edge(" Retriever",  " Writer")

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    print("\n Граф собран\n")
    return graph
