import torch
from torch.utils.data import DataLoader

from constants.common_constants import SPLIT_TEST_NAME
from data import DatasetParent
from utils.experiment_utils.experiment_common_utils import (
    score_functions_selector,
)
from utils.experiment_utils.gradient_descent_utils.checkpoint_utils import load_final_checkpoint
from utils.experiment_utils.gradient_descent_utils.experiment_gradient_descent_common_utils import (
    compute_prediction,
    compute_probability,
    count_parameters,
    generate_report_per_batch,
    loss_function_selector,
)
from utils.experiment_utils.gradient_descent_utils.test_gradient_descent_parent_utils import (
    generate_final_report,
    save_final_results,
)

from ._test_parent import TestParent

__all__ = ["TestGradientDescentParent"]


class TestGradientDescentParent(TestParent):
    def __init__(
        self,
        dataset_directory,
        logger,
        log_init_info,
    ):
        super().__init__(dataset_directory, logger, log_init_info)

    def __prepare_data(self, batch_size, transform=None):
        dataset_test = DatasetParent(
            root=self._dataset_directory,
            split=SPLIT_TEST_NAME,
            transform=transform,
            augmentation=None,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader_test

    def test(
        self,
        model,
        batch_size,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info,
    ):
        self._logger.info("Test normal version")
        self.__test_process_interface_to_support_different_type_of_test(
            self.__test_function,
            model,
            batch_size,
            scores_name_list,
            checkpoint_directory,
            directory_to_save_results,
            directory_to_save_samples,
            test_batch_log_threshold,
            test_batch_sample_threshold,
            log_init_info=log_init_info,
        )

    def ensemble_test(
        self,
        model,
        batch_size,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info,
    ):
        self._logger.info("Test ensemble version")
        self.__test_process_interface_to_support_different_type_of_test(
            self.__ensemble_test_function,
            model,
            batch_size,
            scores_name_list,
            checkpoint_directory,
            directory_to_save_results,
            directory_to_save_samples,
            test_batch_log_threshold,
            test_batch_sample_threshold,
            log_init_info=log_init_info,
        )

    def __test_process_interface_to_support_different_type_of_test(
        self,
        test_function,
        model,
        batch_size,
        scores_name_list,
        checkpoint_directory,
        directory_to_save_results,
        directory_to_save_samples,
        test_batch_log_threshold,
        test_batch_sample_threshold,
        log_init_info,
    ):
        dataloader_test = self.__prepare_data(batch_size, None)
        model_state_dict, loss_function_name = load_final_checkpoint(checkpoint_directory)
        model.load_state_dict(model_state_dict)
        model.to(self._device)
        criterion = loss_function_selector(loss_function_name)
        scores_functions_dict = score_functions_selector(scores_name_list)
        if log_init_info:
            self._logger.info("-" * 80)
            self._logger.info("MODEL ARCHITECTURE")
            self._logger.info("-" * 80)
            self._logger.info("\n%s", model)
            total_params, trainable_params = count_parameters(model, dataloader_test)
            self._logger.info(f"Total parameters:     {total_params:,}")
            self._logger.info(f"Trainable parameters: {trainable_params:,}")
            self._logger.info("-" * 80)

        test_function(
            model,
            dataloader_test,
            criterion,
            scores_functions_dict,
            directory_to_save_results,
            directory_to_save_samples,
            test_batch_log_threshold,
            test_batch_sample_threshold,
        )

    def __test_function(
        self,
        model,
        dataloader_test,
        criterion,
        scores_functions_dict,
        directory_to_save_results,
        directory_to_save_samples,
        test_batch_log_threshold,
        test_batch_sample_threshold,
    ):
        test_batches_num = len(dataloader_test)
        loss_batch_list = []
        labels_list = []
        outputs_logit_list = []
        with torch.no_grad():
            model.eval()
            for batch_index, (eeg_batch, labels_batch) in enumerate(dataloader_test):
                eeg_batch, labels_batch = eeg_batch.to(self._device), labels_batch.to(self._device)
                output_batch_logit = model(eeg_batch)
                loss = criterion(output_batch_logit, labels_batch)
                # loss
                loss_batch_list.append(loss.item())
                # score
                output_batch_logit = output_batch_logit.cpu()
                labels_batch = labels_batch.cpu()
                labels_list.extend(labels_batch)
                outputs_logit_list.extend(output_batch_logit)

                # predictions_batch = compute_prediction(output_batch_logit)
                # probabilities_batch = compute_probability(output_batch_logit)
                # batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
                #     probabilities_batch, labels_batch, scores_functions_dict
                # )
                # # log
                # if test_batch_log_threshold is not None and batch_index % test_batch_log_threshold == 0:
                #     report = generate_report_per_batch(
                #         batch_scores_dict=batch_scores_dict,
                #         batch_loss=loss.item(),
                #         batch_index=batch_index,
                #         batches_num=test_batches_num,
                #         split=SPLIT_TEST_NAME,
                #     )
                #     self._logger.info(report)
                
                # if batch_index % test_batch_sample_threshold == 0:
                #     sample_predicted_label = predictions_batch[0]
                #     channel_importance = compute_integrated_gradients(
                #         model=model,
                #         x=eeg_batch[0],
                #         target_class=1,
                #     )
                #     channel_importance = channel_importance.detach().cpu()
                #     if channel_importance.ndim == 2:
                #         channel_importance = concatenate_hemispharer_based_separated_channels(
                #             channel_importance[0],
                #             channel_importance[1],
                #         )
                #     VisualizerParent.visualize_channels_importance_on_2d_head(
                #         channel_scores=channel_importance,
                #         split=SPLIT_TEST_NAME,
                #         file_name=self._test_sample_index,
                #         montage_name=dataloader_test.dataset.get_montage_name(),
                #         ch_names=dataloader_test.dataset.get_selected_channels(),
                #         true_label=labels_batch[0],
                #         predicted_label=sample_predicted_label,
                #         directory_to_save_figure=directory_to_save_samples,
                #     )
                #     self.__sample_index += 1
        avg_total_loss = sum(loss_batch_list) / test_batches_num
        outputs_logit_list = torch.stack(outputs_logit_list)
        labels_list = torch.stack(labels_list)
        predictions_epoch = compute_prediction(outputs_logit_list)
        probabilities_epoch = compute_probability(outputs_logit_list)
        # epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
        #     probabilities_epoch, labels_list, scores_functions_dict
        # )
        # save_final_results(avg_total_loss, epoch_scores_dict, directory_to_save_results)
        # report = generate_final_report(epoch_scores_dict, avg_total_loss)
        # self._logger.info(report)

    def __ensemble_test_function(
        self,
        model,
        dataloader_test,
        criterion,
        scores_functions_dict,
        directory_to_save_results,
        directory_to_save_samples,
        test_batch_log_threshold,
        test_batch_sample_threshold,
    ):
        test_batches_num = len(dataloader_test)
        loss_batch_list = []
        labels_list = []
        predictions_epoch = []
        outputs_logit_list = []
        with torch.no_grad():
            model.eval()
            for batch_index, (eeg_batch_ensemble, labels_batch) in enumerate(dataloader_test):
                eeg_batch_ensemble = eeg_batch_ensemble.to(self._device)
                labels_batch = labels_batch.to(self._device)
                chunk_size = eeg_batch_ensemble.shape[1]
                output_logit_chunks_batch_list = []
                for chunk_index in range(chunk_size):
                    chunk_batch = eeg_batch_ensemble[:, chunk_index, ...]
                    output_chunk_batch_logit = model(chunk_batch)
                    output_logit_chunks_batch_list.append(output_chunk_batch_logit)
                # loss = compute_chunks_batch_avg_loss(output_logit_chunks_batch_list, labels_batch, criterion)
                # loss_batch_list.append(loss)
                # # score
                # labels_batch = labels_batch.cpu()
                # labels_list.extend(labels_batch)
                # predictions_batch = compute_prediction_by_voting_on_chunks_batch(output_logit_chunks_batch_list)
                # predictions_epoch.extend(predictions_batch)
                # output_batch_logit = compute_avg_column_wise(output_logit_chunks_batch_list)
                # outputs_logit_list.extend(output_batch_logit)

                # probabilities_batch = compute_probability(output_batch_logit)
                # batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
                #     probabilities_batch, labels_batch, scores_functions_dict
                # )
                # if test_batch_log_threshold is not None and batch_index % test_batch_log_threshold == 0:
                #     report = generate_report_per_batch(
                #         batch_scores_dict=batch_scores_dict,
                #         batch_loss=loss,
                #         batch_index=batch_index,
                #         batches_num=test_batches_num,
                #         split=SPLIT_TEST_NAME,
                #     )
                #     self._logger.info(report)
                # if batch_index % test_batch_sample_threshold == 0:
                #     sample_predicted_label = predictions_batch[0]
                #     channel_importance = compute_integrated_gradients(
                #         model=model,
                #         x=eeg_batch_ensemble[0][0],
                #         target_class=1,
                #     )
                #     channel_importance = channel_importance.detach().cpu()
                #     if channel_importance.ndim == 2:
                #         channel_importance = concatenate_hemispharer_based_separated_channels(
                #             channel_importance[0],
                #             channel_importance[1],
                #         )
                #     VisualizerParent.visualize_channels_importance_on_2d_head(
                #         channel_scores=channel_importance,
                #         split=SPLIT_TEST_NAME,
                #         file_name=self._test_sample_index,
                #         montage_name=dataloader_test.dataset.get_montage_name(),
                #         ch_names=dataloader_test.dataset.get_selected_channels(),
                #         true_label=eeg_batch_ensemble[0][0],
                #         predicted_label=sample_predicted_label,
                #         directory_to_save_figure=directory_to_save_samples,
                #     )
                #     self._test_sample_index += 1
        avg_total_loss = sum(loss_batch_list) / test_batches_num
        outputs_logit_list = torch.stack(outputs_logit_list)
        labels_list = torch.stack(labels_list)
        probabilities_epoch = compute_probability(outputs_logit_list)
        # epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
        #     probabilities_epoch, labels_list, scores_functions_dict
        # )
        # save_final_results(avg_total_loss, epoch_scores_dict, directory_to_save_results)
        # report = generate_final_report(epoch_scores_dict, avg_total_loss)
        # self._logger.info(report)
