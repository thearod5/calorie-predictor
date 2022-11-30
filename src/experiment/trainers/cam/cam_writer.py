import json
import os
from typing import Dict, List


class CamLogWriter:
    """
    Responsible for logging and loading state of cam trainer.
    """

    def __init__(self, log_dir_path: str, log_name: str = "kal_log.json", entities_name: str = "logs"):
        """
        Constructs logger with custom attributes.
        :param log_dir_path: Path to directory containing logs.
        :param log_name: The name of the log file.
         :param entities_name: The name of the JSON entities containing logs.
        """
        self.logs: List[Dict] = []
        self._log_path = os.path.join(log_dir_path, log_name)
        self.entities_name: str = entities_name

    def log(self, log: Dict, export: bool = True):
        """
        Adds log to storage.
        :param log: The log to save.
        :param export: Whether to export current logs.
        :return: None
        """
        self.logs.append(log)

    def export(self):
        """
        Saves logs to file.
        :return: None
        """
        payload = {self.entities_name: self.logs}
        with open(self._log_path, "w") as log_file:
            json.dump(payload, log_file, indent=4)
        print("Exported log:", self._log_path)

    def load(self):
        """
        Loads logs if previous run exists.
        :return: None
        """
        with open(self._log_path, "r") as log_file:
            self.logs = json.load(log_file)[self.entities_name]
