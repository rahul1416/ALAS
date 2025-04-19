# import os, json
from pydantic import BaseModel
# from app.core.config import settings

# CHAT_HISTORY_FILE = settings.CHAT_HISTORY_FILE

# def load_chat_history():
#     if os.path.exists(CHAT_HISTORY_FILE):
#         with open(CHAT_HISTORY_FILE, "r") as file:
#             return json.load(file)
#     return {}


# def save_chat_history(chat_history):
#     with open(settings.CHAT_HISTORY_FILE, "w") as file:
#         json.dump(chat_history, file, indent=4)

class QueryRequest(BaseModel):
    query: str
    user_id: str  

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
from app.mongo import db
chat_history_collection = db["chat_history"]

async def load_chat_history(user_id: str):
    """
    Load chat history for a specific user from MongoDB.
    """
    user_doc = await chat_history_collection.find_one({"user_id": user_id})
    if user_doc:
        return user_doc.get("history", [])
    return []

async def save_chat_history(user_id: str, chat_history: list):
    """
    Save chat history for a specific user to MongoDB.
    """
    await chat_history_collection.update_one(
        {"user_id": user_id},
        {"$set": {"history": chat_history}},
        upsert=True
    )