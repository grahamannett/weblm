from typing import List

import cohere
import numpy as np

from .base import Controller, truncate_left


def make_fn(generate_func, tokenize_func, model):
    def func(x):
        if len(x) == 3:
            option, prompt, self = x
            return_likelihoods = "ALL"
        elif len(x) == 4:
            option, prompt, self, return_likelihoods = x

        while True:
            try:
                if len(tokenize_func(prompt)) > 2048:
                    prompt = truncate_left(tokenize_func, prompt)
                return (
                    generate_func(prompt=prompt, max_tokens=0, model=model, return_likelihoods=return_likelihoods)
                    .generations[0]
                    .likelihood,
                    option,
                )
            except cohere.error.CohereError as e:
                print(f"Cohere fucked up: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                continue

    return func


def _generate_func(co_client):
    return co_client.generate


class CohereController(Controller):
    MODEL = "xlarge"
    cohere_client = None

    Controller.exception = cohere.error.CohereError
    Controller.exception_message = "Cohere fucked up11: {0}"

    def __init__(self, co: cohere.Client, objective: str, **kwargs):
        super().__init__(objective)
        self.co = co

        if CohereController.cohere_client is None:
            CohereController.cohere_client = co

        self._fn = make_fn(generate_func=_generate_func(self.co), tokenize_func=self.tokenize, model=self.MODEL)

    def embed(self, texts: List[str], truncate: str = "RIGHT") -> cohere.embeddings.Embeddings:
        return self.co.embed(texts=texts, truncate=truncate)

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.5,
        num_generations: int = 5,
        max_tokens: int = 20,
        stop_sequences: List[str] = ["\n"],
        return_likelihoods: str = "GENERATION",
    ) -> cohere.generation.Generations:

        return _generate_func(self.co)(
            prompt=prompt,
            model=model if model else self.MODEL,
            temperature=temperature,
            num_generations=num_generations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            return_likelihoods="GENERATION",
        )

    def tokenize(self, text: str) -> cohere.tokenize.Tokens:
        return self.co.tokenize(text=text)
