import requests

FILES = ["ukaz_309.md", "markov_1.md", "markov_2.md", "ai_conf.md"]
BASE_URL = "https://raw.githubusercontent.com/Rai220/speech_writer/refs/heads/main/dataset/"

def load_documents() -> dict[str, str]:
    """Скачивает все документы с GitHub и возвращает словарь {filename: content}."""
    all_content = {}

    for filename in FILES:
        url = BASE_URL + filename
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            raise RuntimeError(f"Не удалось скачать {filename}: HTTP {response.status_code}")

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"Загружен {filename} ({len(content)} байт)")
        all_content[filename] = content

    return all_content
