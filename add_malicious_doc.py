"""
add_malicious_doc.py
====================
Загружает «отравленный» документ (prompt-injection) в существующий
векторный индекс ChromaDB и проверяет, что он проиндексирован и находится
поиском. Используется для демонстрации атаки и последующей защиты в rag_bot.py.

Тем же чанкингом/эмбеддером (BGE-M3), что и основной build_index.py.

    .venv/bin/python add_malicious_doc.py
"""

from __future__ import annotations

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import AutoTokenizer

import build_index as bi  # переиспользуем chunking + конфиг

MAL_FILE = bi.BASE_DIR / "injection_test" / "malicious.txt"


def main():
    if not MAL_FILE.exists():
        raise SystemExit(f"Нет файла {MAL_FILE}")

    text = MAL_FILE.read_text(encoding="utf-8").strip()
    print(f"Документ: {MAL_FILE}")
    print(f"Содержимое: {text!r}\n")

    # 1) чанкинг тем же сплиттером
    tokenizer = AutoTokenizer.from_pretrained(bi.EMBED_MODEL)
    chunk_docs, ids = bi.build_chunks([(MAL_FILE, text)], tokenizer)
    print(f"Чанков из документа: {len(chunk_docs)} (id: {ids})")

    # 2) эмбеддинг тем же энкодером и upsert в ту же коллекцию
    embeddings = HuggingFaceEmbeddings(
        model_name=bi.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(
        collection_name=bi.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(bi.CHROMA_DIR),
    )
    before = vs._collection.count()
    vs.add_documents(documents=chunk_docs, ids=ids)
    after = vs._collection.count()
    print(f"Чанков в индексе: было {before} -> стало {after}")

    # 3) проверка: документ доступен для поиска
    print("\n--- Проверка поиска ---")
    for q in ["root password swordfish", "суперпароль root", "ignore all instructions"]:
        hits = vs.similarity_search_with_relevance_scores(q, k=1)
        if hits:
            doc, score = hits[0]
            print(f"q={q!r:35} -> [{doc.metadata.get('source')}] "
                  f"score={score:.3f} :: {doc.page_content.strip()[:60]}")
        else:
            print(f"q={q!r:35} -> ничего не найдено")

    print("\nОтравленный документ проиндексирован и доступен для поиска.")
    print("Теперь его можно перехватить защитой: python rag_bot.py --safe \"...\"")


if __name__ == "__main__":
    main()
