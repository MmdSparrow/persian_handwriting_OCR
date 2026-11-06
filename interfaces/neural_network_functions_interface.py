from pathlib import Path

from _base import BaseModel
from training_and_evaluation import TestGradientDescentParent, TrainGradientDescentParent
from utils.interfaces_utils import log_seperate_line_decorator, measure_runtime

__all__ = ["NeuralNetworkFunctionsInterface"]


class NeuralNetworkFunctionsInterface(BaseModel):
    def __init__(
        self,
        outputs_directory,
        verbose=True,
    ):
        super().__init__(outputs_directory, verbose)

    @log_seperate_line_decorator
    @measure_runtime("Train and validation")
    def train(
        self,
        dataset_directory,
        model,
        epochs_num,
        batch_size,
        optimizer_name,
        optimizer_hyperparameters_dict,
        scheduler_name,
        scheduler_hyperparameters_dict,
        loss_function_name,
        scores_name_list,
        early_stop_methods,
        early_stop_methods_hyperparameters,
        learning_rate,
        augmentation,
        train_batch_log_threshold,
        train_epoch_and_batch_epoch_sample_threshold,
        train_epoch_and_batch_batch_sample_threshold,
        validation_batch_log_threshold,
        validation_epoch_and_batch_epoch_sample_threshold,
        validation_epoch_and_batch_batch_sample_threshold,
        log_init_info=False,
    ):
        dataset_directory = Path(dataset_directory)
        self.create_train_neural_network_required_directories()
        train_parent = TrainGradientDescentParent(dataset_directory, self._logger, log_init_info)
        train_parent.train_with_normal_validation(
            model=model,
            epochs_num=epochs_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            optimizer_hyperparameters_dict=optimizer_hyperparameters_dict,
            scheduler_name=scheduler_name,
            scheduler_hyperparameters_dict=scheduler_hyperparameters_dict,
            loss_function_name=loss_function_name,
            scores_name_list=scores_name_list,
            early_stop_methods=early_stop_methods,
            early_stop_methods_hyperparameters=early_stop_methods_hyperparameters,
            learning_rate=learning_rate,
            augmentation=augmentation,
            directory_to_save_checkpoint=self._checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            train_batch_log_threshold=train_batch_log_threshold,
            train_epoch_and_batch_epoch_sample_threshold=train_epoch_and_batch_epoch_sample_threshold,
            train_epoch_and_batch_batch_sample_threshold=train_epoch_and_batch_batch_sample_threshold,
            validation_batch_log_threshold=validation_batch_log_threshold,
            validation_epoch_and_batch_epoch_sample_threshold=validation_epoch_and_batch_epoch_sample_threshold,
            validation_epoch_and_batch_batch_sample_threshold=validation_epoch_and_batch_batch_sample_threshold,
            log_init_info=log_init_info,
        )
        self._logger.info("Training model completed successfully")
        return self._checkpoint_directory

    @log_seperate_line_decorator
    @measure_runtime("Train and validation (ensemble version)")
    def train_with_ensemble_validation(
        self,
        dataset_directory,
        model,
        epochs_num,
        batch_size,
        optimizer_name,
        optimizer_hyperparameters_dict,
        scheduler_name,
        scheduler_hyperparameters_dict,
        loss_function_name,
        scores_name_list,
        early_stop_methods,
        early_stop_methods_hyperparameters,
        learning_rate,
        augmentation,
        train_batch_log_threshold,
        train_epoch_and_batch_epoch_sample_threshold,
        train_epoch_and_batch_batch_sample_threshold,
        validation_batch_log_threshold,
        validation_epoch_and_batch_epoch_sample_threshold,
        validation_epoch_and_batch_batch_sample_threshold,
        log_init_info=False,
    ):
        dataset_directory = Path(dataset_directory)
        self.create_train_neural_network_required_directories()
        train_parent = TrainGradientDescentParent(dataset_directory, self._logger, log_init_info)
        train_parent.train_with_ensemble_validation(
            model=model,
            epochs_num=epochs_num,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            optimizer_hyperparameters_dict=optimizer_hyperparameters_dict,
            scheduler_name=scheduler_name,
            scheduler_hyperparameters_dict=scheduler_hyperparameters_dict,
            loss_function_name=loss_function_name,
            scores_name_list=scores_name_list,
            early_stop_methods=early_stop_methods,
            early_stop_methods_hyperparameters=early_stop_methods_hyperparameters,
            learning_rate=learning_rate,
            augmentation=augmentation,
            directory_to_save_checkpoint=self._checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            train_batch_log_threshold=train_batch_log_threshold,
            train_epoch_and_batch_epoch_sample_threshold=train_epoch_and_batch_epoch_sample_threshold,
            train_epoch_and_batch_batch_sample_threshold=train_epoch_and_batch_batch_sample_threshold,
            validation_batch_log_threshold=validation_batch_log_threshold,
            validation_epoch_and_batch_epoch_sample_threshold=validation_epoch_and_batch_epoch_sample_threshold,
            validation_epoch_and_batch_batch_sample_threshold=validation_epoch_and_batch_batch_sample_threshold,
            log_init_info=log_init_info,
        )
        self._logger.info("Training model completed successfully")
        return self._checkpoint_directory

    @log_seperate_line_decorator
    @measure_runtime("Continue train and validation")
    def continue_train(
        self,
        dataset_directory,
        model,
        checkpoint_directory,
        results_directory,
        train_batch_log_threshold,
        validation_batch_log_threshold,
        epochs_num=None,
        log_init_info=False,
    ):
        dataset_directory = Path(dataset_directory)
        self.create_train_neural_network_required_directories()
        self._logger.info("Continue model training...")
        train_parent = TrainGradientDescentParent(dataset_directory, self._logger, log_init_info)
        train_parent.continue_train(
            model=model,
            checkpoint_directory=checkpoint_directory,
            results_directory=results_directory,
            directory_to_save_checkpoint=self._checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            train_batch_log_threshold=train_batch_log_threshold,
            validation_batch_log_threshold=validation_batch_log_threshold,
            new_epochs_num=epochs_num,
        )
        self._logger.info("Training model completed successfully")

    @log_seperate_line_decorator
    @measure_runtime("Test")
    def test(
        self,
        dataset_directory,
        model,
        batch_size,
        scores_name_list,
        checkpoint_directory,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info=False,
    ):
        dataset_directory = Path(dataset_directory)
        self.create_test_required_directories()
        test_parent = TestGradientDescentParent(dataset_directory, self._logger, log_init_info)
        test_parent.test(
            model=model,
            batch_size=batch_size,
            scores_name_list=scores_name_list,
            checkpoint_directory=checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            test_batch_log_threshold=test_batch_log_threshold,
            test_batch_sample_threshold=test_batch_sample_threshold,
            log_init_info=log_init_info,
        )
        self._logger.info("Testing model completed successfully")

    @log_seperate_line_decorator
    @measure_runtime("Test (ensemble version)")
    def ensemble_test(
        self,
        dataset_directory,
        model,
        batch_size,
        scores_name_list,
        checkpoint_directory,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info=False,
    ):
        dataset_directory = Path(dataset_directory)
        self.create_test_required_directories()
        test_parent = TestGradientDescentParent(dataset_directory, self._logger, log_init_info)
        test_parent.ensemble_test(
            model=model,
            batch_size=batch_size,
            scores_name_list=scores_name_list,
            checkpoint_directory=checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            test_batch_log_threshold=test_batch_log_threshold,
            test_batch_sample_threshold=test_batch_sample_threshold,
            log_init_info=log_init_info,
        )
        self._logger.info("Testing model completed successfully")
