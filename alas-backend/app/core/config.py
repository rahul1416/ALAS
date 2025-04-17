import os
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=env_path)

class Settings:
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    CHAT_HISTORY_FILE: str = os.getenv("CHAT_HISTORY_FILE")
    FAISS_INDEX: str = os.getenv("FAISS_INDEX")
    MODEL: str = os.getenv("MODEL")
    RERANKER: str = os.getenv("RERANKER")
    TEXT_CHUNKS: str = os.getenv("TEXT_CHUNKS")
    QUESTIONS: str = os.getenv("QUESTIONS")
    MODEL_DIR: str = os.getenv("MODEL_DIR")
    USER_PROFILE: str = os.getenv("USER_PROFILE")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM")
    MONGODB_URI: str = os.getenv("MONGODB_URI")
settings = Settings()
