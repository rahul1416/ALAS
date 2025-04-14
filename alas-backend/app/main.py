from fastapi import FastAPI, Depends, HTTPException
from app.auth import models
from app.auth.auth import create_access_token
from app.auth.schemas import UserLogin
from app.routes import admin, teacher, student

from app.chatbot import chat
from app.ques_ans import quiz

app = FastAPI()

fake_db = {
    "admin1": {"username": "admin1", "password": "adminpass", "role": "admin"},
    "teacher1": {"username": "teacher1", "password": "teacherpass", "role": "teacher"},
    "student1": {"username": "student1", "password": "studentpass", "role": "student"},
}

@app.post("/login")
def login(data: UserLogin):
    user = fake_db.get(data.username)
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token({"username": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer"}

# Register routers
app.include_router(admin.router)
app.include_router(teacher.router)
app.include_router(student.router)
app.include_router(chat.router)
app.include_router(quiz.router)