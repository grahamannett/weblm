from ast import List
import unittest
import unittest.mock as mock
from typing import Callable

from weblm.main import WebLM, WebLMState
from .utils import ScreenshotGrabber


def input_patch(input: str, fn: Callable, *args, **kwargs):
    with mock.patch("builtins.input", return_value=input):
        return fn(*args, **kwargs)


class input_test_list:
    def __init__(self, return_values: list):
        self.return_values = return_values

    def __enter__(self):
        _input_test = input_test(self.return_values.pop(0))
        return _input_test


class input_test:
    def __init__(self, return_value: str):
        self.return_value = return_value
        self.mock = mock.patch("builtins.input", return_value=self.return_value)

    def __enter__(self):
        self.mock.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mock.stop()


class ScreenshotOnFail:
    def __init__(self):
        self.screenshot_grabber = ScreenshotGrabber()

    def __call__(self, state):
        return self.screenshot_grabber(state)


class TestCase(unittest.TestCase):
    # failureException = ScreenshotOnFail(unittest.TestCase.failureException)

    def setUp(self) -> None:
        return super().setUp()

    def test_entry(self):
        commands = [
            "Make a reservation for 2 at 7pm at bistro vida in menlo park",
            "y",
            "y",
        ]

        screenshot_grabber = ScreenshotGrabber()

        weblm = WebLM()

        with input_test(commands.pop(0)):
            state = weblm.start(headless=True)

        self.assertIsNone(state.response, msg=screenshot_grabber(state))

        cmd = commands.pop(0)
        with input_test(cmd) as input:
            state = weblm.step(state)

        screenshot_grabber(state)
        self.assertEqual(state.response, cmd)
        self.assertTrue(len(state.content) > 0)
