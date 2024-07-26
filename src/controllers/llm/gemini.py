# Config gemini api
import os
import dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
dotenv.load_dotenv()


gemini = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.0,
)
