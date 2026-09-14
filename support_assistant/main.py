"""
main.py -- FastAPI app for Module 3 (support_assistant).

Exposes POST /ask which runs the LangGraph graph defined in graph.py and
returns the validated AnswerResponse Pydantic model (answer/sources/
confidence) as JSON.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI
from pydantic import BaseModel

from graph import run_query, AnswerResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="Module 3 of the Zepto Data & AI Platform capstone project.",
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    return run_query(request.query)
