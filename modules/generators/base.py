from abc import ABC, abstractmethod
from modules.baseModule import BaseModule
from typing import List

class BaseGenerator(BaseModule, ABC):
    def __init__(self, name: str = None, max_new_tokens: int = 64, temperature: float = 0.0):
        super().__init__(name)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.last_answer: str | None = None

    @abstractmethod
    def generate(self, query: str, contexts: List[str]) -> str:
        raise NotImplementedError

    def run(self, query: str, contexts: List[str]) -> str:
        ans = self.generate(query, contexts)
        self.last_answer = ans
        return ans