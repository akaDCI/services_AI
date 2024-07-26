from pydantic import BaseModel, Field
import time
import logging
from langchain_core.messages import AIMessage, HumanMessage
from .gemini import gemini
from ._prompt import prompt_template


class LLMInputs(BaseModel):
    question: str = Field(..., description="Question to be answered")
    knowledge: str = Field(None, description="Knowledge to be used")
    item_id: str = Field(None, description="Item ID for retrieve knowledge")
    use_lipsync: bool = Field(False, description="Use lipsync response or not")
    use_audio: bool = Field(False, description="Use audio response or not")


class LLMController:
    def __init__(self):
        self.model = gemini
        self.history = [
            AIMessage(prompt_template)
        ]

    def __add_question(self, question: str, knowledge: str = None):
        if knowledge:
            self.history.append(AIMessage(
                f"The provided knowledge about question is: {knowledge}"))
        self.history.append(HumanMessage(question))

    def __add_answer(self, answer: str):
        self.history.append(answer)

    def generate(self, prompt: str, knowledge: str = None, item_id: str = None):
        # If knowledge is None, fetch the knowledge from the item_id
        # TODO:...

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
