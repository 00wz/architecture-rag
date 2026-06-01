"""
query.py
========
Тестовые поисковые запросы к векторному индексу ChromaDB.

Открывает уже построенный индекс (build_index.py) и для запроса
возвращает top-k наиболее релевантных чанков вместе с метаданными
(источник, имя персонажа, позиция в документе) и score.

Использование:
    # один запрос
    .venv/bin/python query.py "Кто такой Korin Valtaar?"
    .venv/bin/python query.py "Who raised the orphaned Mandalorian-like warrior?" -k 5

    # демо-набор запросов (ru + en), если аргумент не передан
    .venv/bin/python query.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "characters"
EMBED_MODEL = "BAAI/bge-m3"

DEMO_QUERIES = [
    "Кто такой Korin Valtaar и кого он защищал?",
    "Which droid was never given a memory wipe?",
    "Расскажи про пилота истребителя из Corvann",
    "Skarn Master on the High Council during the Accord Era",
]


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def run_query(vs: Chroma, query: str, k: int = 4) -> None:
    print("\n" + "=" * 80)
    print(f"ЗАПРОС: {query}")
    print("=" * 80)
    results = vs.similarity_search_with_relevance_scores(query, k=k)
    if not results:
        print("  (ничего не найдено)")
        return
    for rank, (doc, score) in enumerate(results, 1):
        md = doc.metadata
        print(f"\n[{rank}] score={score:.4f}  | {md.get('character_name')} "
              f"| {md.get('source')} (chunk {md.get('chunk_index')}, "
              f"chars {md.get('char_start')}–{md.get('char_end')})")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 320:
            snippet = snippet[:320] + " …"
        print(f"    {snippet}")


def main():
    parser = argparse.ArgumentParser(description="Search the character KB index")
    parser.add_argument("query", nargs="*", help="Поисковый запрос")
    parser.add_argument("-k", type=int, default=4, help="Сколько чанков вернуть")
    args = parser.parse_args()

    if not CHROMA_DIR.exists():
        raise SystemExit(f"Индекс не найден в {CHROMA_DIR}. Сначала запустите build_index.py")

    vs = get_vectorstore()
    print(f"Чанков в индексе: {vs._collection.count()}")

    if args.query:
        run_query(vs, " ".join(args.query), k=args.k)
    else:
        for q in DEMO_QUERIES:
            run_query(vs, q, k=args.k)


if __name__ == "__main__":
    main()
