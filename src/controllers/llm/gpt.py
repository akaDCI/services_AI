# Config openai api
import os
import dotenv
from langchain_openai import ChatOpenAI
dotenv.load_dotenv()


gpt = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("GPT_API_KEY"),
    temperature=0.0,
)
