#!/usr/bin/env python
#
# natbot.py
#
# Set COHERE_KEY to your API key from os.cohere.ai, and then run this from a terminal.
#

from dataclasses import dataclass
import os
import re
import time
from distutils.util import strtobool
from multiprocessing import Pool
from typing import Any, List, Union

import cohere

# from .controller import Command, Controller, Prompt
from .controllers import Command, Controller, Prompt
from .crawler import URL_PATTERN, Crawler
from .data_saver import CSVSaver as DataSaver


@dataclass
class WebLMState:
    crawler: Crawler
    controller: Controller
    response: Union[Prompt, Command, str]
    content: List[str]


class WebLM:
    def __init__(self):
        self.client = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)
        self.keep_device_ratio = bool(strtobool(os.environ.get("KEEP_DEVICE_RATIO", "False")))
        self.enable_threadpool = bool(strtobool(os.environ.get("ENABLE_TP", "True")))
        self.data_saver = DataSaver()
        self.continue_threshold = 1 / 3

    def get_input(self):
        # helper function that can be wrapped in a mock to test input
        return input()

    def reset(self, headless: bool = False):
        crawler = Crawler(keep_device_ratio=self.keep_device_ratio, headless=headless)

        objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
        print("\nWelcome to WebLM! What is your objective?")
        i = self.get_input()
        if len(i) > 0:
            objective = i

        controller = Controller(self.client, objective, enable_threadpool=self.enable_threadpool)
        return crawler, controller

    def start(self, headless: bool = False):
        crawler, controller = self.reset(headless=headless)
        response = None
        content = []
        crawler.go_to_page("google.com")
        return WebLMState(crawler, controller, response, content)

    def step(self, state: WebLMState = None):

        crawler, controller, response, content = state.crawler, state.controller, state.response, state.content
        if response == "cancel":
            self.data_saver.save_responses(controller.user_responses)
            crawler, controller = self.reset()
        elif response == "success":
            controller.success()
            self.data_saver.save_responses(controller.user_responses)
            exit(0)
        elif response == "back":
            controller.reset_state()
        elif response is not None and re.match(
            f"goto {URL_PATTERN}",
            response,
        ):
            url = re.match(URL_PATTERN, response[5:]).group(0)
            response = None
            crawler.go_to_page(url)
            time.sleep(2)

        content = crawler.crawl()

        while len(content) == 0:
            content = crawler.crawl()
        response = controller.step(crawler.page.url, content, response)

        if isinstance(response, Command):
            crawler.run_cmd(str(response), controller=controller)
            response = None
        elif isinstance(response, Prompt):
            if response.likelihood > self.continue_threshold:
                print(response)
                response = self.get_input()
            else:
                print("continuing with current thing...")
                response = "y"

        return WebLMState(crawler, controller, response, content)


if __name__ == "__main__":
    weblm = WebLM()
    state = weblm.start()
    while True:
        state = weblm.step(state)
