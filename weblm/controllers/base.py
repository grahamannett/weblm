import itertools
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

import numpy as np

from .stubs import *

TYPEABLE = ["input", "select"]
CLICKABLE = ["link", "button"]


def truncate_left(tokenize: Callable, prompt: str, *rest_of_prompt, limit: int = 2048):
    i = 0
    chop_size = 5
    print(f"WARNING: truncating sequence of length {len(tokenize(prompt + ''.join(rest_of_prompt)))} to length {limit}")
    while len(tokenize(prompt + "".join(rest_of_prompt))) > limit:
        prompt = prompt[i * chop_size :]
        i += 1
    return prompt


class Prompt:
    def __init__(self, prompt: str, likelihood: float = 0.0) -> None:
        self.prompt = prompt
        self.likelihood = likelihood

    def __str__(self) -> str:
        return self.prompt

    def __repr__(self) -> str:
        return f"Prompt=>{self.__str__()}"


class Command:
    def __init__(self, cmd: str, likelihood: float = None) -> None:
        self.cmd = cmd
        self.likelihood = likelihood

    def __str__(self) -> str:
        return self.cmd

    def __repr__(self) -> str:
        return f"Command=>{self.__str__()}"


class DialogueState(Enum):
    Unset = None
    Action = "pick action"
    ActionFeedback = "action from feedback"
    Command = "suggest command"
    CommandFeedback = "command from feedback"


def split_list_by_separators(l: List[Any], separator_sequences: List[List[Any]]) -> List[List[Any]]:
    """Split a list by a subsequence.

    split_list_by_separators(range(7), [[2, 3], [5]]) == [[0, 1], [4], [6]]
    """
    split_list: List[List[Any]] = []
    tmp_seq: List[Any] = []

    i = 0
    while i < len(l):
        item = l[i]
        # if this item may be part of one of the separator_sequences
        if any(item == x[0] for x in separator_sequences):
            for s in filter(lambda x: item == x[0], separator_sequences):
                # if we've found a matching subsequence
                if l[i : i + len(s)] == s:
                    if len(tmp_seq) != 0:
                        split_list.append(tmp_seq)
                    tmp_seq = []
                    i += len(s)
                    break
            else:
                i += 1
        else:
            tmp_seq.append(item)
            i += 1

    if len(tmp_seq) != 0:
        split_list.append(tmp_seq)

    return split_list


class Controller:
    """A Cohere-powered controller that takes in a browser state and produces and action.

    The basic outline of this Controller's strategy is:
    1. receive page content from browser
    2. prioritise elements on page based on how relevant they are to the objective
    3. look up similar states from the past
    4. choose between clicking and typing
    5. choose what element to click or what element to type in
    """

    MAX_SEQ_LEN = 2000
    MAX_NUM_ELEMENTS = 50

    client_exception = None
    client_exception_message = "Base Exception Message: {0}"

    def __init__(self, objective: str, enable_threadpool: bool = True):
        """
        Args:
            co (cohere.Client): a Cohere Client
            objective (str): the objective to accomplish
        """
        self.client = None
        self.objective = objective
        self.previous_commands: List[str] = []
        self.moments: List[Tuple[str, str, str]] = []
        self.user_responses: DefaultDict[str, int] = defaultdict(int)
        self.enable_threadpool = enable_threadpool

        self.reset_state()

    def embed(self, texts: List[str], truncate: str = "RIGHT") -> Any:
        raise NotImplementedError("Implement on adapter")

    def generate(self, prompt: str, **kwargs) -> Any:
        raise NotImplementedError("Implement on adapter")

    def tokenize(self, prompt: str) -> Any:
        raise NotImplementedError("Implement on adapter")

    def search(self, query: str, items: List[str], topk: int) -> List[str]:
        embedded_items = np.array(self.embed(texts=items, truncate="RIGHT").embeddings)
        embedded_query = np.array(self.embed(texts=[query], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_query, embedded_items) / (
            np.linalg.norm(embedded_query) * np.linalg.norm(embedded_items, axis=1)
        )
        ind = np.argsort(scores)[-topk:]
        return np.flip(np.array(items)[ind], axis=0)

    def is_running(self):
        return self._step != DialogueState.Unset

    def reset_state(self):
        self._step = DialogueState.Unset
        self._action = None
        self._cmd = None
        self._chosen_elements: List[Dict[str, str]] = []
        self._prioritized_elements = None
        self._pruned_prioritized_elements = None
        self._prioritized_elements_hash = None
        self._page_elements = None

    def success(self):
        for url, elements, command in self.moments:
            self._save_example(url=url, elements=elements, command=command)

    def choose(
        self, func: Callable, template: str, options: List[Dict[str, str]], return_likelihoods: str = "ALL", topk: int = 1
    ) -> List[Tuple[int, Dict[str, str]]]:
        """Choose the most likely continuation of `prompt` from a set of `options`.

        Args:
            template (str): a string template with keys that match the dictionaries in `options`
            options (List[Dict[str, str]]): the options to be chosen from

        Returns:
            str: the most likely option from `options`
        """
        num_options = min(len(options), 64)
        zipped_args = list(zip(options, [template.format(**option) for option in options], [return_likelihoods] * num_options))

        if self.enable_threadpool:
            with ThreadPoolExecutor(num_options) as thread_pool:
                _lh = thread_pool.map(func, zipped_args)

        else:
            _lh = [func(arg) for arg in zipped_args]
        return sorted(_lh, key=lambda x: x[0], reverse=True)[:topk]

    def choose_element(self, template: str, options: List[Dict[str, str]], group_size: int = 10, topk: int = 1) -> List[Dict[str, str]]:
        """A hacky way of choosing the most likely option, while staying within sequence length constraints

        Algo:
        1. chunk `options` into groups of `group_size`
        2. within each group perform a self.choose to get the topk elements (we'll have num_groups*topk elements after this)
        3. flatten and repeat recursively until the number of options is down to topk

        Args:
            template (str): the prompt template with f-string style template tags
            options (List[Dict[str, str]]): a list of dictionaries containing key-value replacements of the template tags
            group_size (int, optional): The size of each group of options to select from. Defaults to 10.
            topk (int, optional): The topk most likely options to return. Defaults to 1.

        Returns:
            List[Dict[str, str]]: The `topk` most likely elements in `options` according to the model
        """
        num_options = len(options)
        num_groups = int(math.ceil(num_options / group_size))

        if num_options == 0:
            raise Exception()

        choices = []
        for i in range(num_groups):
            group = options[i * group_size : (i + 1) * group_size]
            template_tmp = template.replace("elements", "\n".join(item["elements"] for item in group))
            options_tmp = [{"id": item["id"]} for item in group]

            choice = [x[1] for x in self.choose(self._fn, template_tmp, options_tmp, topk=topk)]
            chosen_elements = []
            for x in choice:
                chosen_elements.append(list(filter(lambda y: y["id"] == x["id"], group))[0])
            choices.extend(chosen_elements)

        if len(choices) <= topk:
            return choices
        else:
            return self.choose_element(template, choices, group_size, topk)

    def gather_examples(self, state: str, topk: int = 5) -> List[str]:
        """Simple semantic search over a file of past interactions to find the most similar ones."""
        with open("examples.json", "r") as fd:
            history = json.load(fd)

        if len(history) == 0:
            return []

        embeds = [h["embedding"] for h in history]
        examples = [h["example"] for h in history]
        embeds = np.array(embeds)
        embedded_state = np.array(self.embed(texts=[state], truncate="RIGHT").embeddings[0])
        scores = np.einsum("i,ji->j", embedded_state, embeds) / (np.linalg.norm(embedded_state) * np.linalg.norm(embeds, axis=1))
        ind = np.argsort(scores)[-topk:]
        examples = np.array(examples)[ind]
        return examples

    def _construct_prev_cmds(self) -> str:
        return "\n".join(f"{i+1}. {x}" for i, x in enumerate(self.previous_commands)) if self.previous_commands else "None"

    def _construct_state(self, url: str, page_elements: List[str]) -> str:
        state = state_template
        state = state.replace("$objective", self.objective)
        state = state.replace("$url", url[:100])
        state = state.replace("$previous_commands", self._construct_prev_cmds())
        return state.replace("$browser_content", "\n".join(page_elements))

    def _construct_prompt(self, state: str, examples: List[str]) -> str:
        prompt = prompt_template
        prompt = prompt.replace("$examples", "\n\n".join(examples))
        return prompt.replace("$state", state)

    def _save_example(self, url: str, elements: List[str], command: str):
        state = self._construct_state(url, elements[: self.MAX_NUM_ELEMENTS])
        example = "Example:\n" f"{state}\n" f"Next Command: {command}\n" "----"
        print(f"Example being saved:\n{example}")
        with open("examples.json", "r") as fd:
            history = json.load(fd)
            examples = [h["example"] for h in history]

        if example in examples:
            print("example already exists")
            return

        history.append(
            {
                "example": example,
                "embedding": self.embed(texts=[example]).embeddings[0],
                "url": url,
                "elements": elements,
                "command": command,
            }
        )

        with open("examples_tmp.json", "w") as fd:
            json.dump(history, fd)
        os.replace("examples_tmp.json", "examples.json")

    def _construct_responses(self):
        keys_to_save = ["y", "n", "s", "command", "success", "cancel"]
        responses_to_save = defaultdict(int)
        for key, value in self.user_responses.items():
            if key in keys_to_save:
                responses_to_save[key] = value
            elif key not in keys_to_save and key:
                responses_to_save["command"] += 1

        self.user_responses = responses_to_save
        print(f"Responses being saved:\n{dict(responses_to_save)}")

    def _shorten_prompt(self, url, elements, examples, *rest_of_prompt, target: int = MAX_SEQ_LEN):
        state = self._construct_state(url, elements)
        prompt = self._construct_prompt(state, examples)

        tokenized_prompt = self.tokenize(text=prompt + "".join(rest_of_prompt))
        tokens = tokenized_prompt.token_strings

        split_tokens = split_list_by_separators(tokens, [["EX", "AMP", "LE"], ["Example"], ["Present", " state", ":", "\n"]])
        example_tokens = split_tokens[1:-1]
        length_of_examples = list(map(len, example_tokens))
        state_tokens = split_tokens[-1]
        state_tokens = list(
            itertools.chain.from_iterable(split_list_by_separators(state_tokens, [["----", "----", "----", "----", "--", "\n"]])[1:-1])
        )
        state_tokens = split_list_by_separators(state_tokens, [["\n"]])
        length_of_elements = list(map(len, state_tokens))
        length_of_prompt = len(tokenized_prompt)

        def _fn(i, j):
            state = self._construct_state(url, elements[: len(elements) - i])
            prompt = self._construct_prompt(state, examples[j:])

            return state, prompt

        MIN_EXAMPLES = 1
        i, j = (0, 0)
        while (length_of_prompt - sum(length_of_examples)) + sum(length_of_examples[j:]) > target and j < len(examples) - MIN_EXAMPLES:
            j += 1

        print(f"num examples: {len(examples) - j}")

        state, prompt = _fn(i, j)
        if len(self.tokenize(prompt + "".join(rest_of_prompt))) <= target:
            return state, prompt

        MIN_ELEMENTS = 7
        while (length_of_prompt - sum(length_of_examples[:j]) - sum(length_of_elements)) + sum(
            length_of_elements[: len(length_of_elements) - i]
        ) > target and i < len(elements) - MIN_ELEMENTS:
            i += 1

        print(f"num elements: {len(length_of_elements) - i}")

        state, prompt = _fn(i, j)

        # last resort, start cutting off the bigging of the prompt
        if len(self.tokenize(text=prompt + "".join(rest_of_prompt))) > target:
            prompt = truncate_left(self.tokenize, prompt, *rest_of_prompt, limit=target)

        return state, prompt

    def _generate_prioritization(self, page_elements: List[str], url: str):
        prioritization = prioritization_template
        prioritization = prioritization.replace("$objective", self.objective)
        prioritization = prioritization.replace("$url", url)

        self._prioritized_elements = self.choose(self._fn, prioritization, [{"element": x} for x in page_elements], topk=len(page_elements))
        self._prioritized_elements = [x[1]["element"] for x in self._prioritized_elements]
        self._prioritized_elements_hash = hash(frozenset(page_elements))
        self._pruned_prioritized_elements = self._prioritized_elements[: self.MAX_NUM_ELEMENTS]
        self._step = DialogueState.Action
        print(self._prioritized_elements)

    def pick_action(self, url: str, page_elements: List[str], response: str = None):
        # this strategy for action selection does not work very well, TODO improve this

        if self._step not in [DialogueState.Action, DialogueState.ActionFeedback]:
            return

        state = self._construct_state(url, self._pruned_prioritized_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        if self._step == DialogueState.Action:
            action = " click"
            if any(y in x for y in TYPEABLE for x in page_elements):
                elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE + TYPEABLE), self._pruned_prioritized_elements))
                state, prompt = self._shorten_prompt(url, elements, examples, target=self.MAX_SEQ_LEN)

                actions = self.choose(
                    func=self._fn,
                    template=prompt + "{action}",
                    options=[
                        {
                            "action": " click",
                        },
                        {
                            "action": " type",
                        },
                        {
                            "action": " summary",
                        },
                    ],
                    topk=3,
                )

                # if the model is confident enough, just assume the suggested action is correct
                if (actions[0][0] - actions[1][0]) / -actions[1][0] > 1.0:
                    action = action[0][1]["action"]
                else:
                    action = actions[0][1]["action"]
                    likelihood = [math.exp(a[0]) for a in actions]
                    likelihood = likelihood[0] / sum(likelihood)
                    self._action = action
                    self._step = DialogueState.ActionFeedback
                    return Prompt(eval(f'f"""{user_prompt_1}"""'), likelihood=likelihood)

            self._action = action
            self._step = DialogueState.Command
        elif self._step == DialogueState.ActionFeedback:
            if response == "y":
                pass
            elif response == "n":
                if "click" in self._action:
                    self._action = " type"
                elif "type" in self._action:
                    self._action = " click"
            elif response == "examples":
                examples = "\n".join(examples)
                return Prompt(f"Examples:\n{examples}\n\n" "Please respond with 'y' or 'n'")
            elif re.match(r"search (.+)", response):
                query = re.match(r"search (.+)", response).group(1)
                results = self.search(query, self._page_elements, topk=50)
                return Prompt(f"Query: {query}\nResults:\n{results}\n\n" "Please respond with 'y' or 'n'")
            else:
                return Prompt("Please respond with 'y' or 'n'")

            self._step = DialogueState.Command

    def _get_cmd_prediction(self, prompt: str, chosen_element: str) -> str:
        if "type" in self._action:
            text = None
            while text is None:
                try:
                    num_tokens = 20
                    if len(self.tokenize(prompt)) > 2048 - num_tokens:
                        print(f"WARNING: truncating sequence of length {len(self.tokenize(prompt))}")
                        prompt = truncate_left(self.tokenize, prompt, self._action, chosen_element, limit=2048 - num_tokens)

                    print(len(self.tokenize(prompt + self._action + chosen_element)))
                    text = max(
                        self.generate(
                            prompt=prompt + self._action + chosen_element,
                            model=self.MODEL,
                            temperature=0.5,
                            num_generations=5,
                            max_tokens=num_tokens,
                            stop_sequences=["\n"],
                            return_likelihoods="GENERATION",
                        ).generations,
                        key=lambda x: x.likelihood,
                    ).text
                    print(text)
                except Controller.client_exception as e:
                    print(Controller.client_exception_message.format(e))
                    continue
        else:
            text = ""

        return (self._action + chosen_element + text).strip()

    def use_text(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 192,
        temperature: float = 0.5,
        k: int = 0,
        p: int = 1,
        stop_sequences: List[str] = ["--"],
        return_likelihoods: str = "GENERATION",
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        num_generations: int = 1,
        return_text_only: bool = True,
        **kwargs,
    ):
        response = self.generate(
            prompt=prompt,
            model=model if model else self.MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            k=k,
            p=p,
            frequency_penalty=frequency_penalty,
            num_generations=num_generations,
            presence_penalty=presence_penalty,
            stop_sequences=stop_sequences,
            return_likelihoods=return_likelihoods,
        )

        if return_text_only:
            response = [gen.text for gen in response.generations]

        return response

    def generate_command_choose_element(
        self, url: str, pruned_elements: List[str], examples: np.ndarray, prompt: str, state: str
    ) -> Prompt:
        """Generate a command by choosing an element from the prioritized list"""
        if len(pruned_elements) == 1:
            chosen_element = " " + " ".join(pruned_elements[0].split(" ")[:2])
            self._chosen_elements = [{"id": chosen_element}]
        else:
            state = self._construct_state(url, ["$elements"])
            prompt = self._construct_prompt(state, examples)

            state, prompt = self._shorten_prompt(url, ["$elements"], examples, self._action)

            group_size = 20
            self._chosen_elements = self.choose_element(
                prompt + self._action + "{id}",
                list(map(lambda x: {"id": " " + " ".join(x.split(" ")[:2]), "elements": x}, pruned_elements)),
                group_size,
                topk=5,
            )
            chosen_element = self._chosen_elements[0]["id"]

            state = self._construct_state(url, pruned_elements)
            prompt = self._construct_prompt(state, examples)

            state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)

        cmd = self._get_cmd_prediction(prompt, chosen_element)

        self._cmd = cmd
        self._step = DialogueState.CommandFeedback
        other_options = "\n".join(f"\t({i+2}){self._action}{x['id']}" for i, x in enumerate(self._chosen_elements[1:]))
        return Prompt(eval(f'f"""{user_prompt_2}"""'))

    def generate_command_feedback_handler(
        self, url: str, pruned_elements: List[str], examples: np.ndarray, prompt: str, response: str
    ) -> Union[Command, Prompt]:
        """Handle the feedback from the user on the generated command"""

        match response:
            case "examples":
                examples = "\n".join(examples)
                return Prompt(f"Examples:\n{examples}\n\n" "Please respond with 'y' or 'n'")
            case "prompt":
                chosen_element = self._chosen_elements[0]["id"]
                state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)
                return Prompt(f"{prompt}\n\nPlease respond with 'y' or 's'")
            case "recrawl":
                return Prompt(eval(f'f"""{user_prompt_3}"""'))
            case "elements":
                return Prompt("\n".join(str(d) for d in self._chosen_elements))
            case _:
                if search_match := re.match(r"search (.+)", response):
                    query = search_match.group(1)
                    # results = self.search(query, self._page_elements, topk=50)
                    results = self.search(query=query, items=self._page_elements, topk=50)
                    return Prompt(f"Query: {query}\nResults:\n{results}\n\n" "Please respond with 'y' or 'n'")
                elif re.match(r"\d+", response):
                    chosen_element = self._chosen_elements[int(response) - 1]["id"]
                    state, prompt = self._shorten_prompt(url, pruned_elements, examples, self._action, chosen_element)
                    self._cmd = self._get_cmd_prediction(prompt, chosen_element)
                    if "type" in self._action:
                        return Prompt(eval(f'f"""{user_prompt_3}"""'))
                elif response not in ["y", "s"]:
                    self._cmd = response

        cmd_pattern = r"(click|type) (link|button|input|select) [\d]+( \"\w+\")?"
        other_cmds_pattern = r"(summary)( \"\w+\")?"  # probably want more diverse way to do this, e.g. summarize/summary/etc

        if not re.match(cmd_pattern, self._cmd) and not re.match(other_cmds_pattern, self._cmd):
            return Prompt(f"Invalid command '{self._cmd}'. Must match regex '{cmd_pattern}'. Try again...")

        if response == "s":
            self._save_example(url=url, elements=self._prioritized_elements, command=self._cmd)

        return Command(self._cmd.strip())

    def generate_command(self, url: str, pruned_elements: List[str], response: str = None):
        state = self._construct_state(url, pruned_elements)
        examples = self.gather_examples(state)
        prompt = self._construct_prompt(state, examples)

        match self._step:
            case DialogueState.Command:
                return self.generate_command_choose_element(url, pruned_elements, examples, prompt, state)
            case DialogueState.CommandFeedback:
                if type(cmd := self.generate_command_feedback_handler(url, pruned_elements, examples, prompt, response)) == Prompt:
                    return cmd
            case _:
                cmd = Command(self._cmd.strip())

        self.moments.append((url, self._prioritized_elements, self._cmd))
        self.previous_commands.append(self._cmd)

        self.reset_state()
        return cmd

    def step(self, url: str, page_elements: List[str], response: str = None) -> Union[Prompt, Command]:
        self._step = DialogueState.Action if self._step == DialogueState.Unset else self._step
        self._page_elements = page_elements

        if self._prioritized_elements is None or self._prioritized_elements_hash != hash(frozenset(page_elements)):
            self._generate_prioritization(page_elements, url)

        self.user_responses[response] += 1
        self._construct_responses()
        action_or_prompt = self.pick_action(url, page_elements, response)

        if isinstance(action_or_prompt, Prompt):
            return action_or_prompt

        match self._action.strip():
            case "click":
                pruned_elements = list(filter(lambda x: any(x.startswith(y) for y in CLICKABLE), self._pruned_prioritized_elements))
            case "type":
                pruned_elements = list(filter(lambda x: any(x.startswith(y) for y in TYPEABLE), self._pruned_prioritized_elements))
            case "summary":
                pruned_elements = ["summary of text on page"]

        return self.generate_command(url, pruned_elements, response)
