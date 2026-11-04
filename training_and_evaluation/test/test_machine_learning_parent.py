import numpy as np
from matplotlib import Path

from constants.common_constants import SPLIT_TEST_NAME
from utils.experiment_utils.experiment_common_utils import compute_prediction_based_scores, score_functions_selector
from utils.experiment_utils.machine_learning_utils.checkpoin_utils import load_checkpoint
from utils.experiment_utils.machine_learning_utils.experiment_machine_learning_common_utils import (
    compute_prediction_by_voting_for_chunks,
    generate_report_machine_learning,
)
from utils.load_and_store_dataset_metadata import load_dataset_metadata
from utils.load_and_store_loss_and_scores import store_scores

from ._test_parent import TestParent

__all__ = ["TestMachineLearningParent"]


class TestMachineLearningParent(TestParent):
    def __init__(self, dataset_root, logger, log_init_info):
        super().__init__(dataset_root, logger, log_init_info)

    @staticmethod
    def __read_preprocessed_data_from_input_direcotry(data_directory):
        samples_paths = sorted(Path(eeg_file) for eeg_file in data_directory.glob("*.npy"))
        data_list = []
        label_list = []
        for sample_path in samples_paths:
            data_list.append(np.load(sample_path))
            label_list.append(int(sample_path.stem.split("_")[-1]))
        return np.array(data_list), label_list

    def __prepare_data(self):
        test_data_directory = self.dataset_root / SPLIT_TEST_NAME
        _, self.__montage_name, self.__selected_channels, _, _ = load_dataset_metadata(self.dataset_root)
        x_test, y_test = self.__read_preprocessed_data_from_input_direcotry(test_data_directory)
        return x_test, y_test

    def test(
        self,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
    ):
        self._logger.info("Test normal version")
        self.__test_process_interface_to_support_different_type_of_test(
            validation_function=self.__test_function,
            scores_name_list=scores_name_list,
            checkpoint_directory=checkpoint_directory,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            log_init_info=log_init_info,
        )

    def ensemble_test(
        self,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
    ):
        self._logger.info("Test ensemble version")
        self.__test_process_interface_to_support_different_type_of_test(
            validation_function=self.__ensemble_test_function,
            scores_name_list=scores_name_list,
            checkpoint_directory=checkpoint_directory,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            log_init_info=log_init_info,
        )

    def __test_process_interface_to_support_different_type_of_test(
        self,
        validation_function,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
        # **algorithm_kwargs,
    ):
        model = load_checkpoint(checkpoint_directory)
        if log_init_info:
            self._logger.info("-" * 80)
            self._logger.info(f"MODEL {model.__class__.__name__}")
            self._logger.info(model.get_params())
            self._logger.info("-" * 80)

        scores_functions_dict = score_functions_selector(scores_name_list)
        x_test, y_test = self.__prepare_data()

        # x_test = x_test.reshape(x_test.shape[0], -1)

        validation_function(
            model,
            x_test,
            y_test,
            scores_functions_dict,
            directory_to_save_results,
            directory_to_save_samples,
        )

    def __test_function(
        self,
        model,
        x_test,
        y_test,
        scores_functions_dict,
        directory_to_save_results,
        directory_to_save_samples,
    ):
        y_pred = model.predict(x_test)
        scores_dict = compute_prediction_based_scores(
            predictions=y_pred,
            labels=y_test,
            scores_functions_dict=scores_functions_dict,
        )
        report = generate_report_machine_learning(scores_dict, SPLIT_TEST_NAME)
        self._logger.info(report)
        store_scores(SPLIT_TEST_NAME, scores_dict, directory_to_save_results)

    def __ensemble_test_function(
        self,
        model,
        x_test,
        y_test,
        scores_functions_dict,
        directory_to_save_results,
        directory_to_save_samples,
    ):
        predictions = []
        for x_validation_data in x_test:
            output_logit_chunks_list = []
            for chunk in x_validation_data:
                output_chunk = model.predict(chunk)
                output_logit_chunks_list.append(output_chunk)
            predictions_data = compute_prediction_by_voting_for_chunks(output_logit_chunks_list)
            predictions.extend(predictions_data)
        scores_dict = compute_prediction_based_scores(
            predictions=predictions,
            labels=y_test,
            scores_functions_dict=scores_functions_dict,
        )
        report = generate_report_machine_learning(scores_dict, SPLIT_TEST_NAME)
        self._logger.info(report)
        store_scores(SPLIT_TEST_NAME, scores_dict, directory_to_save_results)
