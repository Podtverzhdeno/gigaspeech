# Speech Writer — Мультиагентный спичрайтер с MCP

Мультиагентная LangGraph-система, которая по теме доклада и его длительности возвращает готовый текст речи.
Релевантные факты и цитаты подбираются из векторного индекса Chroma поверх корпуса markdown-документов.
Доступна как CLI и как **MCP-сервер** (Model Context Protocol).

## Архитектура

```
START → Retriever → Planner → Writer → Critic
              ↑                            │
              └────── retrieve ────────────┤
                                           ├──── fix ──→ Writer
                                           └──── good ──→ END
```

| Агент | Роль | Модель |
|---|---|---|
| Retriever | Ищет релевантные фрагменты в Chroma по теме (1-я итерация) и по тексту речи (далее) | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` |
| Planner | Формирует структуру речи и техническое задание | GigaChat |
| Writer | Пишет/переписывает речь по плану и цитатам из корпуса | GigaChat |
| Critic | Решает: `good` (готово) / `fix` (переписать) / `retrieve` (нужны ещё цитаты) | GigaChat |

Critic ограничен **8 итерациями**, чтобы граф не зацикливался.

## Структура проекта

```
gigaspeech/
├── main.py          # CLI-точка входа (--topic / --duration / --bio)
├── mcp_server.py    # MCP-сервер (FastMCP, stdio)
├── graph.py         # Сборка графа LangGraph
├── agents.py        # Реализация всех 4 агентов
├── state.py         # SpeechWriterState
├── llm.py           # GigaChat + HuggingFace embeddings
├── loader.py        # Загрузка корпуса (локальные файлы либо upstream)
├── vectorstore.py   # Chroma-индекс по всему корпусу
├── config.py        # DEFAULT_TOPIC / DEFAULT_DURATION / build_inputs(...)
├── ukaz_309.md, markov_1.md, markov_2.md, ai_conf.md  # дефолтный корпус
├── requirements.txt
├── pyproject.toml
└── .env.example
```

## Установка

```bash
git clone https://github.com/Podtverzhdeno/gigaspeech.git
cd gigaspeech

# Вариант A — pip
pip install -r requirements.txt

# Вариант B — uv
uv sync

# Заполнить ключ GigaChat
cp .env.example .env
# отредактировать .env: GIGA_CREDENTIALS=<твой access token>
```

## Запуск из CLI

```bash
python main.py --topic "Будущее AI-агентов в Web3" --duration "7 минут"
python main.py --topic "AI в малом бизнесе" --duration "5 минут" --bio "Иван Иванов, CTO ..."
python main.py --topic "..." --duration "..." --output my_speech.md --quiet
```

При первом запуске будет построена векторная БД (5–10 мин на CPU). Последующие запуски используют
`./chroma_db` без переиндексации.

Готовая речь печатается в stdout и сохраняется в `result_speech.md` (или в файл из `--output`).

## Запуск как MCP-сервер

MCP (Model Context Protocol) — открытый протокол, по которому LLM-клиенты (Claude Desktop, Cursor,
LangGraph-агенты, и др.) могут вызывать внешние инструменты. Этот репозиторий предоставляет один
инструмент — `generate_speech_tool(topic, duration, speaker_bio="")`.

### Запуск напрямую (stdio)

```bash
python mcp_server.py
# или, если установлен как пакет:
speech-writer-mcp
```

### Регистрация в Claude Desktop

Открой `~/.config/Claude/claude_desktop_config.json` (Linux) или
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) и добавь:

```json
{
  "mcpServers": {
    "speech-writer": {
      "command": "python",
      "args": ["/абсолютный/путь/к/gigaspeech/mcp_server.py"],
      "env": {
        "GIGA_CREDENTIALS": "<твой access token>"
      }
    }
  }
}
```

После рестарта Claude Desktop в чате появится инструмент `generate_speech_tool` — его можно вызвать
запросом «сгенерируй речь по теме ... на 5 минут».

### Регистрация в Cursor

`Settings → MCP → Add new MCP server`, тип **stdio**, команда: `python /path/to/gigaspeech/mcp_server.py`.

### Проверка через MCP Inspector

```bash
pip install "mcp[cli]"
mcp dev mcp_server.py
```

Откроется веб-инспектор, в котором можно вручную дёрнуть `generate_speech_tool`.

## Корпус документов

В дефолтной поставке индексируются 4 файла:
- `ukaz_309.md` — Указ Президента РФ №309 о нац. целях развития до 2036 года
- `markov_1.md` / `markov_2.md` — Сергей Марков, «Охота на электроовец», тома 1 и 2
- `ai_conf.md` — программа AI Agents BuildCon

Чтобы подменить корпус — положи свои `*.md` рядом с модулями и удали `chroma_db/` (БД пересоберётся
при следующем запуске). Либо используй `loader.load_documents_from_dir(path)` из своего кода.

## Переменные окружения

| Переменная | Назначение |
|---|---|
| `GIGA_CREDENTIALS` | Access token GigaChat (`langchain-gigachat`) |
| `OPENAI_API_KEY` | Зарезервирован, в текущей реализации не используется |

## Результат

CLI пишет речь в `result_speech.md` (или путь из `--output`); MCP-инструмент возвращает её строкой.
