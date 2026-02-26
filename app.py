from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_chroma import get_llm_answer

app = FastAPI(title="Regulatory Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows requests from any domain
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    question: str

@app.post("/api/query")
def ask_question(query: Query):
    answer = get_llm_answer(query.question)
    return {"answer": answer}