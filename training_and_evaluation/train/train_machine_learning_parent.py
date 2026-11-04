import numpy as np
from matplotlib import Path

from constants.common_constants import SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME
from utils.experiment_utils.experiment_common_utils import compute_prediction_based_scores, score_functions_selector
from utils.experiment_utils.machine_learning_utils.checkpoin_utils import save_checkpoint
from utils.experiment_utils.machine_learning_utils.experiment_machine_learning_common_utils import (
    compute_prediction_by_voting_for_chunks,
    generate_report_machine_learning,
)
from utils.load_and_store_dataset_metadata import load_dataset_metadata
from utils.load_and_store_loss_and_scores import store_scores

from ._train_parent import TrainParent

__all__ = ["TrainMachineLearningParent"]


class TrainMachineLearningParent(TrainParent):
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
        train_data_directory = self.dataset_root / SPLIT_TRAIN_NAME
        validation_data_directory = self.dataset_root / SPLIT_VALIDATION_NAME
        _, self.__montage_name, self.__selected_channels, _, _ = load_dataset_metadata(self.dataset_root)
        x_train, y_train = self.__read_preprocessed_data_from_input_direcotry(train_data_directory)
        x_validation, y_validation = self.__read_preprocessed_data_from_input_direcotry(validation_data_directory)
        return x_train, y_train, x_validation, y_validation

    def train_with_normal_validation(
        self,
        model,
        scores_name_list,
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
    ):
        self._logger.info("Train with normal (non ensemble) validation")
        self.__train_process_interface_to_support_different_type_of_validation(
            validation_function=self.__validation_function,
            model=model,
            scores_name_list=scores_name_list,
            directory_to_save_checkpoint=directory_to_save_checkpoint,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            log_init_info=log_init_info,
        )

    def train_with_ensemble_validation(
        self,
        model,
        scores_name_list,
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
    ):
        self._logger.info("Train with ensemble validation")
        self.__train_process_interface_to_support_different_type_of_validation(
            validation_function=self.__ensemble_validation_function,
            model=model,
            scores_name_list=scores_name_list,
            directory_to_save_checkpoint=directory_to_save_checkpoint,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            log_init_info=log_init_info,
        )

    def __train_process_interface_to_support_different_type_of_validation(
        self,
        validation_function,
        model,
        scores_name_list,
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        log_init_info,
        # **algorithm_kwargs,
    ):
        if log_init_info:
            self._logger.info("-" * 80)
            self._logger.info(f"MODEL {model.__class__.__name__}")
            self._logger.info(model.get_params())
            self._logger.info("-" * 80)

        scores_functions_dict = score_functions_selector(scores_name_list)
        x_train, y_train, x_validation, y_validation = self.__prepare_data()

        # x_train = x_train.reshape(x_train.shape[0], -1)
        # x_validation = x_validation.reshape(x_validation.shape[0], -1)

        model.fit(x_train, y_train)

        scores_dict = validation_function(
            model,
            x_validation,
            y_validation,
            scores_functions_dict,
        )

        report = generate_report_machine_learning(scores_dict, SPLIT_VALIDATION_NAME)
        self._logger.info(report)
        save_checkpoint(model, directory_to_save_checkpoint)
        store_scores(SPLIT_VALIDATION_NAME, scores_dict, directory_to_save_results)

    def __validation_function(
        self,
        model,
        x_validation,
        y_validation,
        scores_functions_dict,
    ):
        y_pred = model.predict(x_validation)
        scores_dict = compute_prediction_based_scores(
            predictions=y_pred,
            labels=y_validation,
            scores_functions_dict=scores_functions_dict,
        )
        return scores_dict

    def __ensemble_validation_function(
        self,
        model,
        x_validation,
        y_validation,
        scores_functions_dict,
    ):
        predictions = []
        for x_validation_data in x_validation:
            output_logit_chunks_list = []
            for chunk in x_validation_data:
                output_chunk = model.predict(chunk)
                output_logit_chunks_list.append(output_chunk)
            predictions_data = compute_prediction_by_voting_for_chunks(output_logit_chunks_list)
            predictions.extend(predictions_data)
        scores_dict = compute_prediction_based_scores(
            predictions=predictions,
            labels=y_validation,
            scores_functions_dict=scores_functions_dict,
        )
        return scores_dict
