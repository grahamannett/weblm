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


def print_help():
    print(
        "(g) to visit url\n(u) scroll up\n(d) scroll dow\n(c) to click\n(t) to type\n"
        + "(h) to view commands again\n(r) to run suggested command\n(o) change objective"
    )


@dataclass
class State:
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

    def get_input(self):
        objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
        print("\nWelcome to WebLM! What is your objective?")
        i = input()
        objective = i
        return objective

    def reset(self):
        crawler = Crawler(keep_device_ratio=self.keep_device_ratio)

        objective = "Make a reservation for 2 at 7pm at bistro vida in menlo park"
        print("\nWelcome to WebLM! What is your objective?")
        i = input()
        if len(i) > 0:
            objective = i

        controller = Controller(self.client, objective, enable_threadpool=self.enable_threadpool)
        return crawler, controller

    def start(self):
        crawler, controller = self.reset()
        response = None
        content = []
        crawler.go_to_page("google.com")
        return State(crawler, controller, response, content)

    def run(self, state: State = None):

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
            response = input(str(response))

        return State(crawler, controller, response, content)


if __name__ == "__main__":
    weblm = WebLM()
    state = weblm.start()
    while True:
        state = weblm.run(state)
