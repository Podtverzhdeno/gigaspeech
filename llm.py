import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GIGA_CREDENTIALS = os.getenv("GIGA_CREDENTIALS")

giga = GigaChat(
    model="GigaChat",
    verify_ssl_certs=False,
    profanity_check=False,
    access_token=GIGA_CREDENTIALS,
    streaming=False,
    max_tokens=4096,      # уменьшили с 8000
    temperature=0.7,      # уменьшили с 1 для стабильности
    timeout=900,          # увеличили с 600 до 15 минут
)

# Заменили на более быструю модель (в 3-4 раза быстрее, качество почти такое же)
# all-MiniLM-L6-v2 - компактная и быстрая модель для русского языка
giga_embed = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cpu'},  # Используй 'cuda' если есть GPU
    encode_kwargs={'normalize_embeddings': True}
)