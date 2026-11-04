import time
from functools import wraps

from analizer import (
    AnalyzerFeatureExtractedDataPLI,
    AnalyzerMODMARawAndPreprocessedData,
    AnalyzerMUMTAZRawAndPreprocessedData,
)
from constants.common_constants import (
    MODMA_DATASET_KAGGLE_VERSION_NAME,
    MODMA_DATASET_NAME,
    MUMTAZ_DATASET_KAGGLE_VERSION_NAME,
    MUMTAZ_DATASET_NAME,
    PIPELINE_PLI_NAME,
    PIPELINE_RAW_EEG_NAME,
)
from preprocess import (
    BasicPrepareMODMADataset,
    BasicPrepareMUMTAZDataset,
    BasicPreprocessMODMADataset,
    BasicPreprocessMUMTAZDataset,
    OneTimePreprocessTypedWordsImagesDataset,
    OneTimePreprocessAndFeatureExtractionMUMTAZDataset,
    OneTimePreprocessMODMAataset,
    OneTimePreprocessMUMTAZDataset,
)

__all__ = [
    "basic_preparation_class_selector",
    "basic_preprocess_class_selector",
    "measure_runtime",
    "one_time_preprocess_and_feature_extraction_class_selector",
    "one_time_preprocess_class_selector",
    "processed_data_analyzer_class_selector",
    "raw_data_analyzer_class_selector",
]


def raw_data_analyzer_class_selector(dataset_name):
    if dataset_name == MODMA_DATASET_NAME or dataset_name == MODMA_DATASET_KAGGLE_VERSION_NAME:
        return AnalyzerMODMARawAndPreprocessedData
    if dataset_name == MUMTAZ_DATASET_NAME or dataset_name == MUMTAZ_DATASET_KAGGLE_VERSION_NAME:
        return AnalyzerMUMTAZRawAndPreprocessedData
    raise ValueError()


def processed_data_analyzer_class_selector(pipeline_name):
    if pipeline_name == PIPELINE_PLI_NAME:
        return AnalyzerFeatureExtractedDataPLI
    raise ValueError()


def basic_preparation_class_selector(dataset_name):
    if dataset_name == MODMA_DATASET_KAGGLE_VERSION_NAME:
        return BasicPrepareMODMADataset
    if dataset_name == MUMTAZ_DATASET_NAME or dataset_name == MUMTAZ_DATASET_KAGGLE_VERSION_NAME:
        return BasicPrepareMUMTAZDataset
    raise ValueError()


def basic_preprocess_class_selector(dataset_name):
    if dataset_name == MODMA_DATASET_KAGGLE_VERSION_NAME:
        return BasicPreprocessMODMADataset
    if dataset_name == MUMTAZ_DATASET_NAME or dataset_name == MUMTAZ_DATASET_KAGGLE_VERSION_NAME:
        return BasicPreprocessMUMTAZDataset
    raise ValueError()


def one_time_preprocess_class_selector(dataset_name):
    if dataset_name == MODMA_DATASET_KAGGLE_VERSION_NAME:
        return OneTimePreprocessMODMAataset
    if dataset_name == MUMTAZ_DATASET_NAME or dataset_name == MUMTAZ_DATASET_KAGGLE_VERSION_NAME:
        return OneTimePreprocessMUMTAZDataset
    raise ValueError()


def one_time_preprocess_and_feature_extraction_class_selector(dataset_name):
    if dataset_name == MODMA_DATASET_KAGGLE_VERSION_NAME:
        return OneTimePreprocessTypedWordsImagesDataset
    if dataset_name == MUMTAZ_DATASET_NAME or dataset_name == MUMTAZ_DATASET_KAGGLE_VERSION_NAME:
        return OneTimePreprocessAndFeatureExtractionMUMTAZDataset
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
