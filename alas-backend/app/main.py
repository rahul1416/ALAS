from fastapi import FastAPI, Depends, HTTPException
from app.auth import models
from app.auth.auth import create_access_token
from app.auth.schemas import UserLogin
from app.routes import admin, teacher, student

from app.auth import auth
from app.chatbot import chat
from app.ques_ans import quiz

ap = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

ap.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@ap.get("/")
async def root():
    return {"message": "Welcome to ALAS Backend!"}
    
ap.include_router(auth.router)
ap.include_router(admin.router)
ap.include_router(teacher.router)
ap.include_router(student.router)
ap.include_router(chat.router)
ap.include_router(quiz.router)