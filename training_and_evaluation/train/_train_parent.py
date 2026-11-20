from training_and_evaluation._base_experiment import BaseExperiment


class TrainParent(BaseExperiment):
    def __init__(
        self,
        dataset_directory,
        logger,
        log_init_info,
    ):
        super().__init__(dataset_directory, logger, log_init_info)
        self._train_sample_index = 0
        self._validation_sample_index = 0