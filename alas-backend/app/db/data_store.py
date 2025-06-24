import pandas as pd
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorGridFSBucket
from app.core.config import settings

# Initialize MongoDB client
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client["adaptive_learning"]
questions_collection = db["exaple_questions"]

def upload_csv_to_mongodb(csv_file_path):
    """
    Uploads a CSV file to MongoDB.
    """
    try:
        # Read the CSV file using pandas
        df = pd.read_csv(csv_file_path)

        # Convert the DataFrame to a list of dictionaries (one dictionary per row)
        questions = df.to_dict(orient='records')

        # Insert the questions into MongoDB
        if questions:
            questions_collection.delete_many({})  # Clear existing data (optional)
            questions_collection.insert_many(questions)
            print(f"{len(questions)} questions uploaded successfully.")
        else:
            print("The CSV file is empty.")

    except Exception as e:
        print(f"Error uploading CSV to MongoDB: {str(e)}")

if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = "app/db/corrected_questions.csv"
    upload_csv_to_mongodb(csv_file_path)