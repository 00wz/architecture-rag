"""
app.py
======
REST API (FastAPI) для RAG-бота.

Запуск:
    .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
    # или: .venv/bin/python app.py

Эндпоинты:
    GET  /health           — статус сервиса и число чанков в индексе
    POST /ask  {query,k?}  — задать вопрос боту, получить ответ + источники
    GET  /docs             — авто-документация Swagger UI
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_bot import RagBot

# Бот инициализируется один раз при старте (загрузка эмбеддера + индекса).
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["bot"] = RagBot()
    yield
    STATE.clear()


app = FastAPI(
    title="Character KB RAG Bot",
    description="RAG-бот по базе знаний персонажей (Chroma + BGE-M3 + RetrievalQA, few-shot + CoT)",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(..., description="Вопрос пользователя", examples=["Кто такой Korin Valtaar?"])
    k: int | None = Field(None, description="Сколько чанков извлекать (top-k)")
    safe: bool = Field(
        False,
        description="Включить защиту от prompt-injection (pre-prompt + фильтр "
                    "вредоносных чанков). По умолчанию выключена.",
    )


class Source(BaseModel):
    source: str | None = None
    character_name: str | None = None
    chunk_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    safe_mode: bool


@app.get("/health")
def health():
    bot: RagBot = STATE.get("bot")
    if not bot:
        raise HTTPException(503, "Бот не инициализирован")
    return {
        "status": "ok",
        "embedding_model": "BAAI/bge-m3",
        "chunks_in_index": bot.vectorstore._collection.count(),
        "top_k": bot.retriever.search_kwargs.get("k"),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    bot: RagBot = STATE.get("bot")
    if not bot:
        raise HTTPException(503, "Бот не инициализирован")
    if not req.query.strip():
        raise HTTPException(422, "Пустой запрос")

    # опциональное переопределение top-k на запрос
    if req.k:
        bot.retriever.search_kwargs["k"] = req.k
    try:
        return bot.ask(req.query, safe=req.safe)
    except Exception as e:
        raise HTTPException(500, f"Ошибка генерации: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
