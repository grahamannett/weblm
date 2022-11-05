import csv
import os
from typing import Dict


class CSVSaver:
    def __init__(self, filepath: str = "responses.csv") -> None:
        self.filepath = filepath
        self.keys_to_save = ["y", "n", "s", "command", "success", "cancel"]

    def save_responses(self, user_responses: Dict[str, int]) -> None:
        if not os.path.isfile(self.filepath):
            self._setup_new_file()
        self._append_to_file(user_responses)

    def _setup_new_file(self) -> None:
        with open(self.filepath, "w+") as fd:
            wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
            wr.writerow(self.keys_to_save)

    def _append_to_file(self, user_responses: Dict[str, int]) -> None:
        with open(self.filepath, "a+") as fd:
            wr = csv.writer(fd, quoting=csv.QUOTE_ALL)
            wr.writerow([user_responses[key] for key in self.keys_to_save])
