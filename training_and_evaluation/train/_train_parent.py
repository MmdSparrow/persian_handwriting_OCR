from collections import Counter

from training_and_evaluation._base_experiment import BaseExperiment


class TrainParent(BaseExperiment):
    def __init__(
        self,
        dataset_root,
        logger,
        log_init_info,
    ):
        super().__init__(dataset_root, logger, log_init_info)
        self._train_sample_index = 0
        self._validation_sample_index = 0

        self._total_train_true_predictions_samples = 0
        self._total_train_false_predictions_samples = 0

        self._total_validation_true_predictions_samples = 0
        self._total_validation_false_predictions_samples = 0

        self._first_20_percent_channels_count_train_true_predictions = Counter()
        self._second_20_percent_channels_count_train_true_predictions = Counter()

        self._first_20_percent_channels_count_train_false_predictions = Counter()
        self._second_20_percent_channels_count_train_false_predictions = Counter()

        self._first_20_percent_channels_count_validation_true_predictions = Counter()
        self._second_20_percent_channels_count_validation_true_predictions = Counter()

        self._first_20_percent_channels_count_validation_false_predictions = Counter()
        self._second_20_percent_channels_count_validation_false_predictions = Counter()
