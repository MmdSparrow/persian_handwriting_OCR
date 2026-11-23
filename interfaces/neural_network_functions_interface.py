from _base import BaseModel
from training_and_evaluation import TestGradientDescentParent, TrainGradientDescentParent
from training_and_evaluation.train.neural_network.train_gradient_descent_image_slicing import TrainGradientDescentImageSlicing
from utils.interfaces_utils import dataset_specific_preprocesses_class_selector, log_seperate_line_decorator, measure_runtime

__all__ = ["NeuralNetworkFunctionsInterface"]


class NeuralNetworkFunctionsInterface(BaseModel):
    def __init__(self, dataset_directory, output_directory, seed, verbose=True):
        super().__init__(dataset_directory, output_directory, verbose)
        self.__dataset_name = dataset_directory.name
        self._seed = seed

    @log_seperate_line_decorator
    @measure_runtime("Train and validation")
    def train(
        self,
        chosen_dataset_specific_preprocesses_name_list,
        chosen_dataset_specific_preprocesses_hyperparameters_dict,
        chosen_common_preprocesses_list,
        chosen_common_preprocesses_hyperparameters_dict,
        model,
        epochs_num,
        train_data_files_names_list,
        validation_data_files_names_list,
        batch_size,
        img_splits,
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
        self.create_train_neural_network_required_directories()
        train_parent = TrainGradientDescentImageSlicing(self._dataset_directory, self._outputs_directory, self._logger, self._seed, log_init_info)
        TypedWordsImagesSpecificPreprocesses = dataset_specific_preprocesses_class_selector(self.__dataset_name)
        typed_words_images_specific_preprocesses = TypedWordsImagesSpecificPreprocesses(
            chosen_dataset_specific_preprocesses_name_list,
            chosen_dataset_specific_preprocesses_hyperparameters_dict,
            self._logger,
        )
        train_parent.train(
            dataset_specific_preprocesses=typed_words_images_specific_preprocesses,
            chosen_common_preprocesses_list=chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            model=model,
            epochs_num=epochs_num,
            train_data_files_names_list=train_data_files_names_list,
            validation_data_files_names_list=validation_data_files_names_list,
            batch_size=batch_size,
            img_splits=img_splits,
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
        )
        self._logger.info("Training model completed successfully")
        return self._checkpoint_directory

    # @log_seperate_line_decorator
    # @measure_runtime("Continue train and validation")
    # def continue_train(
    #     self,
    #     model,
    #     checkpoint_directory,
    #     results_directory,
    #     train_batch_log_threshold,
    #     validation_batch_log_threshold,
    #     epochs_num=None,
    #     log_init_info=False,
    # ):
    #     self.create_train_neural_network_required_directories()
    #     self._logger.info("Continue model training...")
    #     train_parent = TrainGradientDescentParent(self._dataset_directory, self._logger, log_init_info)
    #     train_parent.continue_train(
    #         model=model,
    #         checkpoint_directory=checkpoint_directory,
    #         results_directory=results_directory,
    #         directory_to_save_checkpoint=self._checkpoint_directory,
    #         directory_to_save_results=self._results_directory,
    #         directory_to_save_samples=self._samples_directory,
    #         train_batch_log_threshold=train_batch_log_threshold,
    #         validation_batch_log_threshold=validation_batch_log_threshold,
    #         new_epochs_num=epochs_num,
    #     )
    #     self._logger.info("Training model completed successfully")

    @log_seperate_line_decorator
    @measure_runtime("Test")
    def test(
        self,
        model,
        test_data_files_names_list,
        batch_size,
        ensemble_inference_chunk_size,
        scores_name_list,
        checkpoint_directory,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info=False,
    ):
        self.create_test_required_directories()
        test_parent = TestGradientDescentParent(self._dataset_directory, self._logger, log_init_info)
        test_parent.test(
            model=model,
            test_data_files_names_list=test_data_files_names_list,
            batch_size=batch_size,
            ensemble_inference_chunk_size=ensemble_inference_chunk_size,
            scores_name_list=scores_name_list,
            checkpoint_directory=checkpoint_directory,
            directory_to_save_results=self._results_directory,
            directory_to_save_samples=self._samples_directory,
            test_batch_log_threshold=test_batch_log_threshold,
            test_batch_sample_threshold=test_batch_sample_threshold,
        )
        self._logger.info("Testing model completed successfully")
