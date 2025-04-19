from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.core.config import settings

# Initialize MongoDB client
client = AsyncIOMotorClient(settings.MONGODB_URI)

# Access the database and collection
db = client["adaptive_learning"]
profiles_collection = db["profiles"]
users_collection = db['users']
chat_history = db['history']
# Initialize GridFS bucket for storing large files (e.g., model files)
fs = AsyncIOMotorGridFSBucket(db)