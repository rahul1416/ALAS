from fastapi import FastAPI, Depends, HTTPException
from app.auth import models
from app.auth.auth import create_access_token
from app.auth.schemas import UserLogin
from app.routes import admin, teacher, student

from app.auth import auth
from app.chatbot import chat
from app.ques_ans import quiz

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to ALAS Backend!"}
    
app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(teacher.router)
app.include_router(student.router)
app.include_router(chat.router)
app.include_router(quiz.router)