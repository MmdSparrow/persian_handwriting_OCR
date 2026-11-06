import time
from functools import wraps

from constants.common_constants import TYPED_WORDS_IMAGES_DATASET_NAME
from initial_preparation import InitialPreparationTypedWrdsImgesDatasets

__all__ = [
    "initial_preparation_class_selector",
    "log_seperate_line_decorator",
    "measure_runtime",
]


def initial_preparation_class_selector(dataset_name):
    if dataset_name == TYPED_WORDS_IMAGES_DATASET_NAME:
        return InitialPreparationTypedWrdsImgesDatasets
    raise ValueError()


def measure_runtime(description):
    def measutre_runtime_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_time = int(time.perf_counter() - start_time)
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                self = args[0]
                self._logger.info(f"{description} runtime: {runtime}")

        return wrapper

    return measutre_runtime_decorator


def log_seperate_line_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]
        result = func(*args, **kwargs)
        self._logger.info("=" * 363)
        return result

    return wrapper
