from training_and_evaluation._base_experiment import BaseExperiment


class TestParent(BaseExperiment):
    def __init__(
        self,
        dataset_directory,
        logger,
        log_init_info,
    ):
        super().__init__(dataset_directory, logger, log_init_info)
        self._test_sample_index = 0
