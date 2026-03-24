from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from llm import giga_embed

COLLECTION_NAME = "speech_writer"

BOOKS = {
    "markov_1.md": "Сергей Марков. Охота на электроовец. Большая книга искусственного интеллекта. Том 1",
    "markov_2.md": "Сергей Марков. Охота на электроовец. Большая книга искусственного интеллекта. Том 2",
}

def build_vectorstore(all_content: dict[str, str]) -> Chroma:
    """
    Чанкирует книги Маркова и загружает их в Chroma с батчингом для Gigachat.
    Метаданные содержат название книги для последующего цитирования.
    """
    Chroma(collection_name=COLLECTION_NAME, embedding_function=giga_embed).delete_collection()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # уменьшили размер чанка
        chunk_overlap=0,
        length_function=len,
        keep_separator=True,
    )

    BATCH_SIZE = 10
    vectorstore = None
    total_chunks = 0

    for filename, book_title in BOOKS.items():
        chunks = splitter.split_text(all_content[filename])
        metadatas = [{"document": book_title} for _ in chunks]
        print(f" {book_title}: {len(chunks)} чанков")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch_texts = chunks[i:i + BATCH_SIZE]
            batch_metadatas = metadatas[i:i + BATCH_SIZE]

            if vectorstore is None:
                vectorstore = Chroma.from_texts(
                    batch_texts,
                    embedding=giga_embed,
                    metadatas=batch_metadatas,
                    collection_name=COLLECTION_NAME,
                )
            else:
                vectorstore.add_texts(batch_texts, metadatas=batch_metadatas)

            total_chunks += len(batch_texts)

    print(f"Векторная БД готова: {total_chunks} документов")
    return vectorstore