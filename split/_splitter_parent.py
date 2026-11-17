class SplitterParent:
    def __init__(self, logger, seed):
        self._logger = logger
        self._seed = seed

    def _log_preparation_details(self, test_ratio, validation_ratio):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Train ratio={1.0 - (test_ratio + validation_ratio)}, Validation ratio={validation_ratio}")

    def _log_preparation_details_fold(self, test_ratio, folds_num):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Folds number={folds_num}")
