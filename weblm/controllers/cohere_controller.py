from typing import Callable, List

import cohere
from requests.exceptions import ConnectionError


from weblm.controllers.controller import Controller, truncate_left

AVAILABLE_MODELS = ["command-xlarge-20221108", "xlarge"]
CLIENT_EXCEPTION = cohere.error.CohereError
CLIENT_EXCEPTION_MESSAGE = "Cohere fucked up: {0}"


def make_fn(generate_func, tokenize_func, model):
    """helper to make func for threadpool"""

    def _fn(x):
        """func that is actually called by threadpool

        this takes a prompt and returns the likelihood of that prompt (hence max_tokens=0)
        """
        if len(x) == 2:
            option, prompt = x
            return_likelihoods = "ALL"
        elif len(x) == 3:
            option, prompt, return_likelihoods = x

        while True:
            try:
                if len(tokenize_func(prompt)) > 2048:
                    prompt = truncate_left(tokenize_func, prompt)
                response = generate_func(prompt=prompt, max_tokens=0, model=model, return_likelihoods=return_likelihoods)
                return (response.generations[0].likelihood, option)
            except cohere.error.CohereError as e:
                print(f"Cohere fucked up: {e}")
                continue
            except ConnectionError as e:
                print(f"Connection error: {e}")
                continue

    return _fn


def _generate_func(co_client: cohere.Client) -> Callable:
    return co_client.generate


def _tokenize_func(co_client: cohere.Client) -> Callable:
    return co_client.tokenize


class CohereController(Controller):
    MODEL = "command-xlarge-20221108"
    cohere_client = None  # make it possible to get client without instantiating controller

    Controller.client_exception = CLIENT_EXCEPTION
    Controller.client_exception_message = CLIENT_EXCEPTION_MESSAGE

    def __init__(self, co: cohere.Client, objective: str, **kwargs):
        super().__init__(objective, **kwargs)
        self.client = co

        if CohereController.cohere_client is None:
            CohereController.cohere_client = co

        self._fn = make_fn(generate_func=self.generate, tokenize_func=self.tokenize, model=self.MODEL)

    def embed(self, texts: List[str], truncate: str = "RIGHT") -> cohere.embeddings.Embeddings:
        return self.client.embed(texts=texts, truncate=truncate)

    def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.5,
        num_generations: int = 5,
        max_tokens: int = 20,
        stop_sequences: List[str] = None,
        return_likelihoods: str = "GENERATION",
        **kwargs,
    ) -> cohere.generation.Generations:
        return self.client.generate(
            prompt=prompt,
            model=model if model else self.MODEL,
            temperature=temperature,
            num_generations=num_generations,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            return_likelihoods=return_likelihoods,
            **kwargs,
        )

    def tokenize(self, text: str) -> cohere.tokenize.Tokens:
        return self.client.tokenize(text=text)
