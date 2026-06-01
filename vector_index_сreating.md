# Создание векторного индекса базы знаний персонажей

## Используемый код

| Файл | Назначение |
|------|------------|
| `build_index.py` | Чанкинг → эмбеддинги → загрузка в ChromaDB (создаёт или обновляет индекс) |
| `query.py` | Тестовые поисковые запросы к индексу (top-k чанков + метаданные + score) |
| `requirements.txt` | Зафиксированные зависимости |

## Результат

| Файл | Назначение |
|------|------------|
| `chroma_db/` | **Готовый индекс** (персистентный ChromaDB, `chroma.sqlite3`) |
| `index_stats.json` | Статистика последней сборки |

### Метаданные каждого чанка
`source` (относит. путь), `file_path`, `file_name`, `character_name` / `title`,
`chunk_id`, `chunk_index`, `n_chunks_in_doc`, `char_start`, `char_end`,
`word_count`, `token_count` — для цитирования и указания позиции в источнике.

## Установка

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
# CPU-сборка torch при необходимости:
# .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Сборка индекса

```bash
.venv/bin/python build_index.py            # создать / обновить (upsert по chunk_id)
.venv/bin/python build_index.py --reset    # пересоздать коллекцию с нуля
```

Идемпотентно: чанки записываются с детерминированными `id` (`<file>::chunk_<i>`),
повторный запуск обновляет существующие записи, новые — добавляет.

## Тестирование:

```bash
.venv/bin/python query.py "Кто такой Korin Valtaar и кого он защищал?"
.venv/bin/python query.py "Which droid was never given a memory wipe?" -k 5
.venv/bin/python query.py            # демо-набор ru/en запросов
```

