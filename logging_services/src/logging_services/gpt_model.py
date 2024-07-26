from openai import AzureOpenAI as az
from .prompt import PROMPT_REPORT
 
class OpenAIClient:
    def __init__(self, azure_endpoint, api_key, api_version="2024-05-13"):
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.client = az(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )
 
    def call_openai(self, data_info, task) -> str:
      prompt = PROMPT_REPORT.replace("{data_info}", data_info).replace("{task}", task)
      completion = self.client.chat.completions.create(
            model="GPT4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
      )
      message_openai = completion.choices[0].message.content.lstrip("\n")
      message_openai = message_openai.replace("\n", "")
      return message_openai