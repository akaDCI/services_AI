from pydantic import BaseModel, Field
import enum
import time
import logging
from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from ._prompt import prompt_template
from .gemini import gemini
from .gpt import gpt


PROVIDERS = {
    "gemini": gemini,
    "gpt": gpt
}


class LLMProvider(enum.Enum):
    gemini = "gemini"
    gpt = "gpt"


class LLMInputs(BaseModel):
    question: str = Field(..., description="Question to be answered")
    knowledge: str = Field(None, description="Knowledge to be used")
    use_stream: bool = Field(False, description="Use stream response or not")
    provider: LLMProvider = Field(
        LLMProvider.gemini, description="LLM provider")


class LLMController:
    def __init__(self):
        self.model = self.__get_provider(LLMProvider.gemini)
        self.history = [
            AIMessage(prompt_template)
        ]

    def __get_provider(self, provider: LLMProvider):
        global PROVIDERS
        if provider.value in PROVIDERS:
            return PROVIDERS[provider.value]
        else:
            raise ValueError(f"Provider {provider} not found")

    def set_provider(self, provider: LLMProvider):
        self.model = self.__get_provider(provider)

    def __add_question(self, question: str, knowledge: str = None):
        if knowledge:
            self.history.append(AIMessage(
                f"The provided knowledge about question is: {knowledge}"))
        self.history.append(HumanMessage(question))

    def __add_answer(self, answer: AIMessage):
        self.history.append(answer)

    def __add_answer_chunk(self, chunk: AIMessageChunk):
        last_message = self.history[-1]
        if isinstance(last_message, AIMessageChunk):
            last_message.content = chunk.content
        else:
            self.history.append(chunk)

    def generate(self, prompt: str, knowledge: str):
        # Add the question to the history
        self.__add_question(prompt, knowledge)

        # Generate the answer
        _s = time.time()
        answer = self.model.invoke(self.history)
        logging.info(
            f"Generated answer successfull [{round(time.time() - _s, 2)}s]")

        # Add the answer to the history
        self.__add_answer(answer)
        return answer.content

    def generate_async(self, prompt: str, knowledge: str):
        # Add the question to the history
        self.__add_question(prompt, knowledge)

        # Generate the answer
        _s = time.time()
        streamer = self.model.stream(self.history)
        logging.info(
            f"Generated answer successfull [{round(time.time() - _s, 2)}s]")

        # Generate the answer
        for chunk in streamer:
            self.__add_answer_chunk(chunk)
            yield chunk.content
