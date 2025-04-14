import os, json
from pydantic import BaseModel
from app.core.config import settings

CHAT_HISTORY_FILE = settings.CHAT_HISTORY_FILE

def load_chat_history():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, "r") as file:
            return json.load(file)
    return {}


def save_chat_history(chat_history):
    with open(settings.CHAT_HISTORY_FILE, "w") as file:
        json.dump(chat_history, file, indent=4)

class QueryRequest(BaseModel):
    query: str
    user_id: str  