"""
rag_bot.py
==========
RAG-бот по базе знаний персонажей вымышленной вселенной.

Возможности:
  * загрузка готового векторного индекса ChromaDB (build_index.py);
  * эмбеддинг запроса ТЕМ ЖЕ энкодером (BAAI/bge-m3), что и при индексации;
  * поиск ближайших чанков (Chroma similarity search);
  * формирование промпта с найденными фрагментами;
  * Few-shot prompting — 1-2 примера, РЕАЛЬНО извлечённые из базы знаний;
  * Chain-of-Thought — system-промпт заставляет модель рассуждать по шагам;
  * генерация ответа готовой цепочкой LangChain RetrievalQA;
  * честный ответ «Я не знаю», если в контексте нет информации.

LLM (url / model-id / api-key) читается из переменных окружения:
    LLM_BASE_URL   — базовый URL OpenAI-совместимого API (напр. https://api.openai.com/v1)
    LLM_MODEL      — id модели (напр. gpt-4o-mini)
    LLM_API_KEY    — ключ API
необязательные:
    LLM_TEMPERATURE (default 0.0), RAG_TOP_K (default 4)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "characters"
EMBED_MODEL = "BAAI/bge-m3"          # тот же энкодер, что и при построении индекса

TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# --------------------------------------------------------------------------- #
# System-промпт с Chain-of-Thought
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """Ты — помощник-эксперт по вымышленной вселенной (персонажи, \
планеты, организации). Ты отвечаешь на русском языке СТРОГО на основе \
предоставленного контекста — фрагментов из базы знаний.

Правила:
1. Сначала размышляй, потом отвечай. Всегда выписывай свои шаги рассуждения.
2. Опирайся ТОЛЬКО на факты из блока «Контекст». Ничего не выдумывай.
3. Если в контексте нет информации для ответа — честно напиши «Я не знаю» \
и не пытайся угадать.
4. В конце указывай источники (имена файлов из метаданных контекста).

Формат ответа строго такой:
Рассуждение:
1. <первый шаг>
2. <второй шаг>
3. <вывод из контекста>
Ответ: <краткий итоговый ответ ИЛИ «Я не знаю»>
Источники: <файлы-источники или «—»>"""

# --------------------------------------------------------------------------- #
# Few-shot примеры (извлечены из РЕАЛЬНОЙ базы знаний: файлы 0.txt и 18.txt).
# Демонстрируют модели нужный CoT-формат, в т.ч. кейс «Я не знаю».
# --------------------------------------------------------------------------- #
FEWSHOT_BLOCK = """Ниже — примеры того, как нужно отвечать.

Пример 1
Контекст:
[0.txt] Korin Valtaar ... An orphan born on Qen Vatra ... was raised on Selvath \
as a foundling by the Keepers of the Vigil, an orthodox religious sect ...
Q: Кто воспитал осиротевшего Korin Valtaar?
A:
Рассуждение:
1. В вопросе спрашивают, кто воспитал Korin Valtaar.
2. В контексте сказано, что он был сиротой и его вырастили «the Keepers of the Vigil» на планете Selvath.
3. Значит, воспитателями были Keepers of the Vigil.
Ответ: Korin Valtaar воспитали Хранители Бдения (Keepers of the Vigil) на планете Selvath.
Источники: 0.txt

Пример 2
Контекст:
[18.txt] A6-N2 ... was an A6-series astromech droid manufactured by Cybertech \
Forgeworks ... A6-N2 was never given a full memory wipe nor did he ever receive new programming ...
Q: Какой компанией был произведён дроид A6-N2?
A:
Рассуждение:
1. Нужно найти производителя дроида A6-N2.
2. В контексте указано, что он «manufactured by Cybertech Forgeworks».
3. Следовательно, производитель — Cybertech Forgeworks.
Ответ: Дроид A6-N2 произведён компанией Cybertech Forgeworks.
Источники: 18.txt

Пример 3 (нет ответа в контексте)
Контекст:
[18.txt] A6-N2 ... was an A6-series astromech droid ...
Q: Какое любимое блюдо у Korin Valtaar?
A:
Рассуждение:
1. Спрашивают про любимое блюдо Korin Valtaar.
2. В контексте говорится только про дроида A6-N2, про еду и про Valtaar ничего нет.
3. Данных для ответа нет.
Ответ: Я не знаю.
Источники: —"""

# Финальный пользовательский ход: контекст + вопрос
HUMAN_TEMPLATE = """{fewshot}

Теперь ответь на реальный вопрос в том же формате.

Контекст:
{context}

Q: {question}
A:"""


# --------------------------------------------------------------------------- #
# Сборка бота
# --------------------------------------------------------------------------- #
def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Переменная окружения {name} не задана. "
            f"Нужны: LLM_BASE_URL, LLM_MODEL, LLM_API_KEY (см. .env.example)."
        )
    return val


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> Chroma:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"Индекс не найден в {CHROMA_DIR}. Сначала запустите build_index.py."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=_require_env("LLM_BASE_URL"),
        model=_require_env("LLM_MODEL"),
        api_key=_require_env("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        timeout=60,
    )


def build_prompt() -> ChatPromptTemplate:
    """Чат-промпт: system (CoT) + human (few-shot + контекст + вопрос).

    RetrievalQA (chain_type='stuff') подставит найденные чанки в {context},
    а вопрос пользователя — в {question}. {fewshot} фиксируем как partial.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)]
    )
    return prompt.partial(fewshot=FEWSHOT_BLOCK)


class RagBot:
    """Обёртка над цепочкой RetrievalQA с few-shot + CoT."""

    def __init__(self, top_k: int = TOP_K):
        self.embeddings = get_embeddings()
        self.vectorstore = get_vectorstore(self.embeddings)
        self.llm = get_llm()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        # каждый найденный чанк попадает в {context} с префиксом-источником,
        # чтобы модель могла корректно указать файлы в строке «Источники».
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="[{source}] {page_content}",
        )
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": build_prompt(),
                "document_prompt": document_prompt,
            },
        )

    def ask(self, question: str) -> dict:
        result = self.chain.invoke({"query": question})
        sources = []
        for d in result.get("source_documents", []):
            md = d.metadata
            sources.append(
                {
                    "source": md.get("source"),
                    "character_name": md.get("character_name"),
                    "chunk_id": md.get("chunk_id"),
                    "char_start": md.get("char_start"),
                    "char_end": md.get("char_end"),
                }
            )
        return {
            "question": question,
            "answer": result["result"].strip(),
            "sources": sources,
        }


# --------------------------------------------------------------------------- #
# CLI: .venv/bin/python rag_bot.py "вопрос"
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Использование: python rag_bot.py "ваш вопрос"')
        raise SystemExit(1)

    bot = RagBot()
    out = bot.ask(" ".join(sys.argv[1:]))
    print("\n" + out["answer"])
    print("\n--- Источники ---")
    for s in out["sources"]:
        print(f"  {s['character_name']} | {s['source']} | {s['chunk_id']}")
