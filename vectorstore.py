"""Сборка векторного индекса по корпусу markdown-документов.

Индексирует ВСЕ переданные документы (а не только книги Маркова), чтобы Retriever мог
находить релевантные цитаты для произвольной темы доклада.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import giga_embed

COLLECTION_NAME = "speech_writer"
PERSIST_DIR = "./chroma_db"

# Человекочитаемые названия источников. Если файла нет в этом словаре —
# в качестве названия используется имя файла без расширения.
SOURCE_TITLES: dict[str, str] = {
    "markov_1.md": "Сергей Марков. Охота на электроовец. Том 1",
    "markov_2.md": "Сергей Марков. Охота на электроовец. Том 2",
    "ukaz_309.md": "Указ Президента РФ №309 о национальных целях развития до 2036 г.",
    "ai_conf.md": "Программа конференции AI Agents BuildCon",
}

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200
BATCH_SIZE = 32


def _title_for(filename: str) -> str:
    return SOURCE_TITLES.get(filename, Path(filename).stem)


def build_vectorstore(
    all_content: dict[str, str],
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Chroma:
    """Чанкирует все документы из корпуса и строит/переиспользует Chroma-индекс."""
    if os.path.exists(persist_dir):
        print(f"Загружаем готовую векторную БД из {persist_dir}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=giga_embed,
            persist_directory=persist_dir,
        )
        try:
            count = vectorstore._collection.count()
        except Exception:
            count = "?"
        print(f"Векторная БД загружена ({count} документов)")
        return vectorstore

    print("Создаём новую векторную БД...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        keep_separator=True,
    )

    vectorstore: Chroma | None = None
    total_chunks = 0

    for filename, content in all_content.items():
        title = _title_for(filename)
        chunks = splitter.split_text(content)
        metadatas = [{"document": title, "source": filename} for _ in chunks]
        print(f"{title}: {len(chunks)} чанков")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_texts = chunks[i : i + BATCH_SIZE]
            batch_metadatas = metadatas[i : i + BATCH_SIZE]

            if vectorstore is None:
                vectorstore = Chroma.from_texts(
                    batch_texts,
                    embedding=giga_embed,
                    metadatas=batch_metadatas,
                    collection_name=collection_name,
                    persist_directory=persist_dir,
                )
            else:
                vectorstore.add_texts(batch_texts, metadatas=batch_metadatas)

            total_chunks += len(batch_texts)
            print(f"  обработано {total_chunks} чанков")

    if vectorstore is None:
        raise RuntimeError("Корпус пуст — нечего индексировать")

    print(f"Векторная БД готова: {total_chunks} чанков (сохранена в {persist_dir})")
    return vectorstore
