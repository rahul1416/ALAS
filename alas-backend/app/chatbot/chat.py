from fastapi import APIRouter, HTTPException
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from datetime import datetime
from app.core.config import settings
import numpy as np
import pickle
import faiss
import os

from app.chatbot.views import load_chat_history, save_chat_history, QueryRequest

load_dotenv()

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Load models and index once
llm = ChatGroq(model_name=settings.MODEL)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
reranker = CrossEncoder(settings.RERANKER)
faiss_index = faiss.read_index(settings.FAISS_INDEX)

with open(settings.TEXT_CHUNKS, "rb") as f:
    text_chunks = pickle.load(f)

@router.post("/query")
async def answer_query(request: QueryRequest):
    query = request.query
    user_id = request.user_id
    chat_history = load_chat_history()

    if user_id not in chat_history:
        chat_history[user_id] = []

    chat_history[user_id].append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })

    last_msgs = chat_history[user_id][-5:]
    history_context = "\n".join(
        f"{m['role'].capitalize()} ({m['timestamp']}): {m['content']}" for m in last_msgs
    )

    query_embedding = np.array(embeddings.embed_query(query)).astype("float32").reshape(1, -1)
    _, indices = faiss_index.search(query_embedding, 5)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    retrieved_docs = [Document(page_content=c) for c in retrieved_chunks]

    if not retrieved_docs:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    # Rerank
    query_doc_pairs = [(query, doc.page_content) for doc in retrieved_docs]
    scores = reranker.predict(query_doc_pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]
    top_docs = sorted_docs[:3]

    prompt = ChatPromptTemplate.from_template(
        "Answer this question technically & thoroughly.\n\n"
        "Chat History:\n{history}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Your Answer:"
    )

    qa_chain = create_stuff_documents_chain(llm, prompt)
    response = qa_chain.invoke({
        "history": history_context,
        "context": top_docs,
        "question": query
    })

    bot_msg = {
        "role": "bot",
        "content": response,
        "timestamp": datetime.now().isoformat()
    }
    chat_history[user_id].append(bot_msg)
    save_chat_history(chat_history)

    return {"response": response, "history": chat_history[user_id]}


@router.get("/history/{user_id}")
async def get_history(user_id: str):
    chat_history = load_chat_history()
    return {"history": chat_history.get(user_id, [])}
