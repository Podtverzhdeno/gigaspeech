import os
from dotenv import load_dotenv
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_openai.chat_models import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GIGA_CREDENTIALS = os.getenv("GIGA_CREDENTIALS")

giga = GigaChat(
    model="GigaChat-Max",
    verify_ssl_certs=False,
    profanity_check=False,
    access_token=GIGA_CREDENTIALS,
    streaming=False,
    max_tokens=8000,
    temperature=1,
    timeout=600,
)

chatgpt = ChatOpenAI(api_key=GIGA_CREDENTIALS, model="gpt-4o") if OPENAI_API_KEY else None

giga_embed = GigaChatEmbeddings(
    model="EmbeddingsGigaR",
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False,
    access_token=GIGA_CREDENTIALS,
)