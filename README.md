# Speech Writer — ReAct-агент с MCP

Один **ReAct-агент** на `langchain.agents.create_agent` (LangChain 1.x), который по теме доклада и его
длительности возвращает готовый текст речи. Агент сам решает, какие инструменты и в каком порядке
вызывать; из векторного индекса Chroma поверх корпуса markdown-документов он подтягивает релевантные
цитаты. Доступен как CLI и как **MCP-сервер** (Model Context Protocol).

## Архитектура

```
                        ┌──────────────────────────┐
  user msg  ─────────►  │   ReAct-агент (GigaChat) │  ◄── system prompt
                        └──────┬───────────────────┘
                               │ tool_calls
                               ▼
            ┌────────────────────────────────────────────┐
            │ retrieve_quotes  → Chroma RAG              │
            │ plan_speech      → LLM                     │
            │ write_speech     → LLM + self-verify цитат │
            │ verify_quotes    → детерминированно        │
            │ critique_speech  → LLM (APPROVED)          │
            └────────────────────────────────────────────┘
                               │ tool_results
                               ▼
                       AIMessage  ──► finish (Markdown речь)
```

| Инструмент | Аргументы | Что делает |
|---|---|---|
| `retrieve_quotes` | `query` | similarity_search по Chroma, кладёт цитаты в state |
| `plan_speech` | `topic, duration, speaker_bio?` | LLM-план + ТЗ, читает цитаты из state |
| `write_speech` | `topic, duration, speaker_bio?, critique?` | LLM-речь в Markdown; **внутри сама проверяет, что все цитаты есть в корпусе, и переписывает себя до 2 раз, удаляя выдуманные источники** |
| `verify_quotes` | — | детерминированный re-check последней речи; вернёт `OK` или `FAKE: <список>` |
| `critique_speech` | `topic, duration` | LLM-критика последней речи; вернёт `APPROVED` или текст замечаний |

Цикл «думать → инструмент → результат → думать» крутится самим агентом, пока модель не выдаст
финальный ответ без `tool_calls`. Подстраховка от зацикливания — `recursion_limit=40` (≈20 вызовов
инструментов).

### Почему длинные строки не передаются через аргументы

GigaChat при tool-call'е с длинным аргументом-строкой (например, целый текст речи) подставляет
плейсхолдер вроде «писавший текст выше» вместо реального содержимого. Чтобы не зависеть от этого,
длинные артефакты — найденные цитаты, текущий план, последняя версия речи — хранятся в модульном
`_STATE` внутри `tools.py`. `retrieve_quotes` пишет туда найденные цитаты, `write_speech` пишет
туда последнюю речь; `verify_quotes` и `critique_speech` читают оттуда. Агент в `tool_calls`
передаёт только короткие значения: тему, длительность, биографию, критику.

`reset_state()` вызывается в начале каждого `generate_speech(...)`, поэтому соседние сессии не
пересекаются.

## Структура проекта

```
gigaspeech/
├── main.py          # CLI-точка входа (--topic / --duration / --bio)
├── mcp_server.py    # MCP-сервер (FastMCP, stdio)
├── graph.py         # build_agent() через create_react_agent
├── tools.py         # 5 инструментов ReAct-агента + state
├── llm.py           # GigaChat + HuggingFace embeddings
├── loader.py        # Загрузка корпуса
├── vectorstore.py   # Chroma-индекс по всему корпусу
├── config.py        # DEFAULT_TOPIC / DEFAULT_DURATION
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
python main.py --topic "..." --duration "..." --recursion-limit 60   # агенту больше шагов
```

При первом запуске будет построена векторная БД (5–10 мин на CPU). Последующие запуски используют
`./chroma_db` без переиндексации.

В консоли видно, какие инструменты вызывал агент:

```
[1s] model: AIMessage → tool_calls: ['retrieve_quotes']
[1s] tools: ToolMessage → (Цитата из «Сергей Марков. Охота на электроовец. Том 1»): ...
[2s] model: AIMessage → tool_calls: ['plan_speech']
[6s] model: AIMessage → tool_calls: ['write_speech']
[10s] model: AIMessage → tool_calls: ['critique_speech']
[11s] tools: ToolMessage → APPROVED
[14s] model: AIMessage → # AI-агенты в Web3 ...
```

Готовая речь печатается в stdout и сохраняется в `result_speech.md` (или в файл из `--output`).

## Запуск как MCP-сервер

MCP (Model Context Protocol) — открытый протокол, по которому LLM-клиенты (Claude Desktop, Cursor,
LangGraph-агенты, и др.) могут вызывать внешние инструменты. Этот репозиторий предоставляет один
инструмент — `generate_speech_tool(topic, duration, speaker_bio="")`, который запускает
ReAct-агента и возвращает готовую речь.

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
