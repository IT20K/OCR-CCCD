import os
from dotenv import load_dotenv

load_dotenv() # take environment variables from .env.

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/quickroom_db")
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ('true', '1', 't') 