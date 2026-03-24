# Speech Writer — Мультиагентный спичрайтер

Генерирует тексты публичных выступлений по методу Билла Шмарцо (ex-CTO Dell).

## Архитектура

```
START →  Planner →  Writer →  Critic
                          ↑               ↓
                      Retriever ←── (retrieve)
                                          ↓
                                         END
```

| Агент | Роль | Модель |
|---|---|---|
|  Planner | Анализирует тему, биографию, Указ №309 → структура + ТЗ | GigaChat-Max |
|  Writer | Пишет речь по ТЗ с учётом критики и цитат | GPT-4o |
|  Critic | Оценивает речь: `good` / `fix` / `retrieve` | GigaChat-Max |
|  Retriever | Ищет цитаты из книг Маркова (RAG + Chroma) | GigaChat Embeddings |

## Структура проекта

```
speech_writer/
├── main.py          # Точка входа
├── graph.py         # Сборка LangGraph графа
├── agents.py        # Все 4 агента
├── state.py         # SpeechWriterState (TypedDict)
├── llm.py           # Инициализация GigaChat и GPT-4o
├── loader.py        # Загрузка документов с GitHub
├── vectorstore.py   # Chroma + GigaChat Embeddings
├── config.py        # Биография спикера и входные данные
├── requirements.txt
└── .env.example
```

## Запуск

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env из шаблона и заполнить ключи
cp .env.example .env

# 3. Запустить
python main.py
```

## Документы (загружаются автоматически)

- `ukaz_309.md` — Указ Президента №309 о нац. целях до 2036
- `markov_1.md` — Сергей Марков. Охота на электроовец. Том 1
- `markov_2.md` — Сергей Марков. Охота на электроовец. Том 2
- `ai_conf.md` — Информация о конференции AI Agents BuildCon

## Результат

Речь сохраняется в `result_speech.md`
