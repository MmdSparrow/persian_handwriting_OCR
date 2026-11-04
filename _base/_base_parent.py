import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from constants.base_directories_names_constants import (
    LOGS_DIRECTORY_NAME,
    OUTPUTS_MAIN_DIRECTORY_NAME,
)


class BaseParent:
    def __create_output_directory(self, output_directory):
        if output_directory is None:
            self._output_directory_name = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")  # noqa: DTZ002
            self._output_directory = OUTPUTS_MAIN_DIRECTORY_NAME / self._output_directory_name
        else:
            self._output_directory = Path(output_directory)

        self._log_directory = self._output_directory / LOGS_DIRECTORY_NAME
        os.makedirs(self._log_directory, exist_ok=True)

    def __init_logger(self, verbose):
        self._logger = logging.getLogger("core_logger")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            file_handler = logging.FileHandler(self._log_directory / "runtime.log")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            if verbose:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)

    def __init__(self, verbose=True, output_directory=None):
        self.__create_output_directory(output_directory)
        self.__init_logger(verbose)
