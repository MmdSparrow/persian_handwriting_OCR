__all__ = ["PreprocessParent"]


class PreprocessParent:
    __EXCLUDED_FEATURE_EXTRACTION_KWARGS = ["montage_positions", "channels_names"]  # noqa: RUF012
    __LEVEL_TWO_FEATURE_EXTRACTION_PARAMETERS = ["epoch_duration", "augmentation_overlap", "ensemble_batch_num"]  # noqa: RUF012

    def __init__(self, logger, seed):
        self._logger = logger
        self._seed = seed

    def _log_preparation_details(self, test_ratio, validation_ratio):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Train ratio={1.0 - (test_ratio + validation_ratio)}, Validation ratio={validation_ratio}")

    def _log_preparation_details_fold_version(self, test_ratio, folds_num):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Folds number={folds_num}")

    def _log_info(self, persian_alphabet_characters):
        self._logger.info("persian alphabet characters:")
        self._logger.info(persian_alphabet_characters)

    def _log_dataset_info(self):
        pass

    def _log_common_preprocesses_info(self, chosen_common_preprocesses_list):
        pass
