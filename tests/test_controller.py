import unittest
import os

import cohere

from weblm.controllers.controller import DialogueState
from weblm.controllers import Controller

objective = "buy bodywash"
url = "https://www.google.com/?gws_rd=ssl"
page_elements = [
    "button 0 L3eUgb",
    'link 1 "About"',
    'link 2 "Store"',
    'link 3 "Gmail"',
    'link 4 "Images"',
    "button 5 gb_F gb_ld gb_l gb_Ff gbwa",
    'button 6 class="gb_d" aria-label="Google apps"',
    'link 7 "Sign in"',
    "button 9 iblpc",
    "input 10 gLFyf q text combobox Search Search",
    'button 11 class="XDyW0e" aria-label="Search by voice"',
    'button 12 aria-label="Search by image" alt="Camera search"',
    'button 13 value="Google Search" aria-label="Google Search" name="btnK"',
    'button 14 value="I\'m Feeling Lucky" aria-label="I\'m Feeling Lucky" name="btnI"',
    'link 15 "Advertising"',
    'link 16 "Business"',
    'link 17 "How Search works"',
    'link 18 "Carbon neutral since 2007"',
    'link 19 "Privacy"',
    'link 20 "Terms"',
    'button 21 "Settings"',
]


_step = DialogueState.Action
_pruned_prioritized_elements = [
    "input 10 gLFyf q text combobox Search Search",
    'link 3 "Gmail"',
    'button 14 value="I\'m Feeling Lucky" aria-label="I\'m Feeling Lucky" name="btnI"',
    'link 7 "Sign in"',
    'button 13 value="Google Search" aria-label="Google Search" name="btnK"',
    'link 2 "Store"',
    'link 1 "About"',
    'link 15 "Advertising"',
    'link 4 "Images"',
    'link 19 "Privacy"',
    'link 16 "Business"',
    'button 21 "Settings"',
    'button 12 aria-label="Search by image" alt="Camera search"',
    'link 20 "Terms"',
    'link 17 "How Search works"',
    "button 9 iblpc",
    "button 0 L3eUgb",
    'link 18 "Carbon neutral since 2007"',
    'button 6 class="gb_d" aria-label="Google apps"',
    'button 11 class="XDyW0e" aria-label="Search by voice"',
    "button 5 gb_F gb_ld gb_l gb_Ff gbwa",
]


class TestPickAction(unittest.TestCase):
    def reset_controller(self):
        self.controller._step = _step
        self.controller._pruned_prioritized_elements = _pruned_prioritized_elements

    def test_pick_action(self):
        client = cohere.Client(os.environ.get("COHERE_KEY"), check_api_key=False)
        controller = Controller(client, objective, enable_threadpool=False)
        self.controller = controller

        self.reset_controller()

        generation = self.controller.generate_pick_action(url=url, page_elements=page_elements)
        breakpoint()

        # action = controller.pick_action(url=url, page_elements=page_elements, response=None, topk_examples=1)
        # print(action.metadata["action"], action.likelihood)
        # self.reset_controller()
        # action = controller.pick_action(url=url, page_elements=page_elements, response=None, topk_examples=3)
        # # print(action)
        # print(action.metadata["action"], action.likelihood)
        # self.reset_controller()
        # action = controller.pick_action(url=url, page_elements=page_elements, response=None, topk_examples=5)
        # print(action.metadata["action"], action.likelihood)

        # # test how n exmaples helps

        # breakpoint()
