"""
build_index.py
================
Строит векторный индекс ChromaDB по базе знаний персонажей.

Пайплайн:
  1. Читает все *.txt из knowledge_base/ (один файл = один персонаж).
  2. Режет тексты на чанки RecursiveCharacterTextSplitter'ом
     (длина считается в токенах токенизатором BGE-M3).
  3. Генерирует эмбеддинги многоязычной моделью BAAI/bge-m3.
  4. Складывает чанки + эмбеддинги + метаданные в персистентный ChromaDB.
     Индекс создаётся, если его нет, иначе обновляется (upsert по id).

Результат полностью совместим с LangChain `Chroma` + `RetrievalQA`,
которые будут использованы в прототипе RAG-бота.

Запуск:
    .venv/bin/python build_index.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "characters"

EMBED_MODEL = "BAAI/bge-m3"          # многоязычная (ru/en) модель эмбеддингов
CHUNK_SIZE_TOKENS = 500              # ~целевой размер чанка в токенах (500–1000)
CHUNK_OVERLAP_TOKENS = 80           # перекрытие, чтобы не рвать контекст


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #
def extract_character_name(text: str) -> str:
    """Достаёт имя персонажа из первого предложения.

    Тексты построены по шаблону "<Имя>, also known as ..." или
    "<Имя> was a ...", поэтому берём фрагмент до первой запятой
    или до глагола-связки (was/were/is/are) — что встретится раньше.
    """
    head = text.strip().split("\n", 1)[0]
    # позиция первой запятой
    comma = head.find(",")
    # позиция первого глагола-связки
    m = re.search(r"\b(was|were|is|are)\b", head)
    verb = m.start() if m else -1

    candidates = [p for p in (comma, verb) if p != -1]
    cut = min(candidates) if candidates else min(len(head), 60)
    name = head[:cut].strip(" \t\"',.")
    return name or head[:60].strip()


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- #
# Основной пайплайн
# --------------------------------------------------------------------------- #
def load_documents() -> list[tuple[Path, str]]:
    files = sorted(
        glob.glob(str(KB_DIR / "*.txt")),
        key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else Path(p).stem,
    )
    docs = []
    for f in files:
        text = Path(f).read_text(encoding="utf-8").strip()
        if text:
            docs.append((Path(f), text))
    return docs


def build_chunks(docs, tokenizer):
    """Режем каждый документ на чанки и формируем list[Document] с метаданными."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    )

    chunk_docs: list[Document] = []
    ids: list[str] = []

    for path, text in docs:
        character = extract_character_name(text)
        pieces = splitter.split_text(text)

        # позиция чанка в исходном документе (символьный offset)
        search_from = 0
        for idx, piece in enumerate(pieces):
            start = text.find(piece, search_from)
            if start == -1:                       # на случай нормализации пробелов
                start = text.find(piece)
            end = start + len(piece) if start != -1 else -1
            search_from = end if end != -1 else search_from

            n_tokens = len(tokenizer.encode(piece, add_special_tokens=False))
            chunk_id = f"{path.stem}::chunk_{idx}"

            metadata = {
                "source": str(path.relative_to(BASE_DIR)),
                "file_path": str(path),
                "file_name": path.name,
                "character_name": character,
                "title": character,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "n_chunks_in_doc": len(pieces),
                "char_start": start,
                "char_end": end,
                "word_count": word_count(piece),
                "token_count": n_tokens,
            }
            chunk_docs.append(Document(page_content=piece, metadata=metadata))
            ids.append(chunk_id)

    return chunk_docs, ids


def main():
    parser = argparse.ArgumentParser(description="Build ChromaDB index for the character KB")
    parser.add_argument("--reset", action="store_true",
                        help="Удалить существующую коллекцию перед загрузкой")
    args = parser.parse_args()

    t0 = time.perf_counter()

    print(f"[1/4] Загрузка документов из {KB_DIR} ...")
    docs = load_documents()
    print(f"      Найдено документов: {len(docs)}")

    print(f"[2/4] Чанкинг (RecursiveCharacterTextSplitter, {CHUNK_SIZE_TOKENS} ток., "
          f"overlap {CHUNK_OVERLAP_TOKENS}) ...")
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    chunk_docs, ids = build_chunks(docs, tokenizer)
    print(f"      Создано чанков: {len(chunk_docs)}")

    print(f"[3/4] Инициализация модели эмбеддингов {EMBED_MODEL} (CPU) ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if args.reset:
        print("      --reset: очищаю существующую коллекцию ...")
        try:
            vectorstore.delete_collection()
        except Exception as e:
            print(f"      (нечего удалять: {e})")
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    print(f"[4/4] Генерация эмбеддингов и запись в ChromaDB ({CHROMA_DIR}) ...")
    t_embed = time.perf_counter()
    # add_documents с фиксированными id => upsert: создаёт или обновляет
    vectorstore.add_documents(documents=chunk_docs, ids=ids)
    embed_time = time.perf_counter() - t_embed

    total_in_index = vectorstore._collection.count()
    total_time = time.perf_counter() - t0

    stats = {
        "embedding_model": EMBED_MODEL,
        "collection": COLLECTION_NAME,
        "persist_directory": str(CHROMA_DIR),
        "n_documents": len(docs),
        "n_chunks_built": len(chunk_docs),
        "n_chunks_in_index": total_in_index,
        "chunk_size_tokens": CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "embedding_time_sec": round(embed_time, 2),
        "total_time_sec": round(total_time, 2),
    }
    (BASE_DIR / "index_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== ГОТОВО =====")
    for k, v in stats.items():
        print(f"  {k:22}: {v}")
    print(f"\nСтатистика сохранена в {BASE_DIR / 'index_stats.json'}")


if __name__ == "__main__":
    main()
