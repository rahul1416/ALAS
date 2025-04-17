# app/db/mongo.py
from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

# Initialize MongoDB client
client = AsyncIOMotorClient(settings.MONGODB_URI)

# Access the database and collection
db = client["loginAlas"]  # Replace with your database name
users_collection = db["users"]  # Replace with your collection name