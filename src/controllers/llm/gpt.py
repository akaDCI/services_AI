# Config openai api
import os
import dotenv
from langchain_openai import AzureChatOpenAI
dotenv.load_dotenv()


os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"


gpt = AzureChatOpenAI(
    deployment_name="fsa-gpt4o",
    model_version="2024-05-13",
)
