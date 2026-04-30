"""Загрузка корпуса документов для RAG.

По умолчанию использует локальные markdown-файлы рядом с модулями (если они есть);
при отсутствии — скачивает их из upstream-репозитория. Локальные файлы НЕ перезаписываются.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_FILES: tuple[str, ...] = (
    "ukaz_309.md",
    "markov_1.md",
    "markov_2.md",
    "ai_conf.md",
)
BASE_URL = "https://raw.githubusercontent.com/Rai220/speech_writer/refs/heads/main/dataset/"

REPO_ROOT = Path(__file__).resolve().parent


def _download(filename: str, dest: Path) -> None:
    url = BASE_URL + filename
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Не удалось скачать {filename}: HTTP {response.status_code}")
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def load_documents(
    files: tuple[str, ...] = DEFAULT_FILES,
    docs_dir: str | os.PathLike[str] | None = None,
) -> dict[str, str]:
    """Возвращает словарь {filename: content} для всех документов корпуса.

    Поведение:
    - Если файл уже существует локально — читаем как есть, не качаем.
    - Если файла нет — пробуем скачать из upstream-репозитория и кладём рядом.
    """
    base = Path(docs_dir) if docs_dir else REPO_ROOT
    base.mkdir(parents=True, exist_ok=True)

    all_content: dict[str, str] = {}
    for filename in files:
        path = base / filename
        if not path.exists():
            print(f"Скачиваем {filename} из upstream...")
            _download(filename, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"Загружен {filename} ({len(content)} символов)")
        all_content[filename] = content

    return all_content


def load_documents_from_dir(docs_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Грузит все *.md файлы из указанной директории. Полезно для произвольного корпуса."""
    base = Path(docs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Каталог не найден: {base}")
    all_content: dict[str, str] = {}
    for path in sorted(base.glob("*.md")):
        with open(path, "r", encoding="utf-8") as f:
            all_content[path.name] = f.read()
        print(f"Загружен {path.name} ({len(all_content[path.name])} символов)")
    if not all_content:
        raise RuntimeError(f"В {base} не найдено ни одного *.md файла")
    return all_content
