import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from visualizer import VisualizerParent
from constants.common_constants import (
    ACCURACY_LIMIT,
    CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
    CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY,
    EARLY_STOP_ACCURACY_LIMIT,
    EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    SCHEDULER_PLATEAU_NAME,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)
from data import DatasetParent
from info.vocab_info import VOCAB_LETTERS_INDEXES_DICT
from utils.character_index_converter import letter_to_index_converter
from utils.experiment_utils.experiment_common_utils import (
    # compute_prediction_based_scores,
    # compute_probability_based_scores,
    score_functions_selector,
    # store_channel_importance_counts,
)
from utils.experiment_utils.gradient_descent_utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.experiment_utils.gradient_descent_utils.experiment_gradient_descent_common_utils import (
    compute_prediction,
    compute_probability,
    count_parameters,
    generate_report_per_batch,
    loss_function_selector,
)
from utils.experiment_utils.gradient_descent_utils.image_slicing_utils import get_one_hot_start_token, get_start_token
from utils.experiment_utils.gradient_descent_utils.train_gradient_descent_parent_utils import (
    FinalCheckpointDetails,
    generate_final_checkpoint_report,
    generate_report_per_epoch,
    optimizer_selector,
    save_final_results,
    scheduler_selector,
)
from utils.load_and_store_dataset_preprocess_info import store_dataset_preprocesses_info

from .._train_parent import TrainParent

__all__ = ["TrainGradientDescentImageSlicing"]


class TrainGradientDescentImageSlicing(TrainParent):
    def __init__(self, dataset_directory, outputs_directory, logger, seed, log_init_info):
        super().__init__(dataset_directory, outputs_directory, logger, seed, log_init_info)
        if log_init_info:
            self._logger.info("If model has `gradient_clip`, `weight_clip` or `initialize_weights` methods, they are called automatically")

    def __prepare_data(
        self,
        train_data_files_names_list,
        validation_data_files_names_list,
        dataset_specific_preprocesses,
        chosen_common_preprocesses_list,
        chosen_common_preprocesses_hyperparameters_dict,
        batch_size,
        augmentation,
    ):
        self._log_chosen_preprocesses_info(
            chosen_preprocesses_name_list=dataset_specific_preprocesses._chosen_dataset_specific_preprocesses_name_list,
            chosen_preprocesses_hyperparameters_dict=dataset_specific_preprocesses._chosen_dataset_specific_preprocesses_hyperparameters_dict,
            preprocesses_name_list_in_order_of_apply=dataset_specific_preprocesses._CHOSEN_DATASET_SPECIFIC_PREPROCESSES_NAME_LIST_IN_ORDER_OF_APPLY,
            preprocesses_and_their_hyperparameters_dict=dataset_specific_preprocesses._CHOSEN_DATASET_SPECIFIC_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
            preprocesses_type="dataset-specific",
        )
        self._log_chosen_preprocesses_info(
            chosen_preprocesses_name_list=chosen_common_preprocesses_list,
            chosen_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            preprocesses_name_list_in_order_of_apply=CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY,
            preprocesses_and_their_hyperparameters_dict=CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
            preprocesses_type="common",
        )
        store_dataset_preprocesses_info(
            chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict,
            [self._dataset_directory, self._outputs_directory],
        )

        dataset_train = DatasetParent(
            dataset_directory=self._dataset_directory,
            data_files_names_list=train_data_files_names_list,
            dataset_specific_preprocesses=dataset_specific_preprocesses,
            chosen_common_preprocesses_list=chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            augmentation=augmentation,
            seed=self._seed,
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset_validation = DatasetParent(
            dataset_directory=self._dataset_directory,
            data_files_names_list=validation_data_files_names_list,
            dataset_specific_preprocesses=dataset_specific_preprocesses,
            chosen_common_preprocesses_list=chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            augmentation=None,
            seed=self._seed,
        )
        dataloader_validation = DataLoader(
            dataset_validation,
            batch_size=batch_size,
            shuffle=True,  # for better sampling
            # shuffle=False,
        )
        return dataloader_train, dataloader_validation

    def train(
        self,
        dataset_specific_preprocesses,
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
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        train_batch_log_threshold,
        train_epoch_and_batch_epoch_sample_threshold,
        train_epoch_and_batch_batch_sample_threshold,
        validation_batch_log_threshold,
        validation_epoch_and_batch_epoch_sample_threshold,
        validation_epoch_and_batch_batch_sample_threshold,
    ):
        if early_stop_methods is None:
            early_stop_methods = []
        train_epoch_and_batch_sample_threshold = None
        if train_epoch_and_batch_epoch_sample_threshold is not None and train_epoch_and_batch_batch_sample_threshold is not None:
            train_epoch_and_batch_sample_threshold = (
                train_epoch_and_batch_epoch_sample_threshold,
                train_epoch_and_batch_batch_sample_threshold,
            )
        validation_epoch_and_batch_sample_threshold = None
        if validation_epoch_and_batch_epoch_sample_threshold is not None and validation_epoch_and_batch_batch_sample_threshold is not None:
            validation_epoch_and_batch_sample_threshold = (
                validation_epoch_and_batch_epoch_sample_threshold,
                validation_epoch_and_batch_batch_sample_threshold,
            )
        model.to(self._device)
        optimizer = optimizer_selector(optimizer_name, model, learning_rate, optimizer_hyperparameters_dict)
        criterion = loss_function_selector(loss_function_name)
        scores_functions_dict = score_functions_selector(scores_name_list)
        scheduler = scheduler_selector(scheduler_name, optimizer, scheduler_hyperparameters_dict)
        final_checkpoint_details = FinalCheckpointDetails()
        min_loss = np.inf
        loss_list_train_epoch = []
        scores_list_train_epoch = []
        loss_list_validation_epoch = []
        scores_list_validation_epoch = []
        dataloader_train, dataloader_validation = self.__prepare_data(
            train_data_files_names_list=train_data_files_names_list,
            validation_data_files_names_list=validation_data_files_names_list,
            dataset_specific_preprocesses=dataset_specific_preprocesses,
            chosen_common_preprocesses_list=chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            batch_size=batch_size,
            augmentation=augmentation,
        )
        if self._log_init_info:
            self._logger.info("-" * 80)
            self._logger.info(f"Epochs: {epochs_num}")
            self._logger.info(f"Batch Size: {batch_size}")
            self._logger.info(f"Loss Fucntion: {loss_function_name}")
            self._logger.info(f"Optimizer: {optimizer_name}")
            self._logger.info(f"Optimizer Hyperparameters: {optimizer_hyperparameters_dict}")
            self._logger.info(f"Learning Rate: {learning_rate}")
            self._logger.info(f"Scheduler: {scheduler_name}")
            self._logger.info(f"Scheduler Hyperparameters: {scheduler_hyperparameters_dict}")
            self._logger.info(f"Early Stop Methods: {early_stop_methods}")
            self._logger.info(f"Early Stop Methods Hyperparameters: {early_stop_methods_hyperparameters}")
            self._logger.info("-" * 80)
            self._logger.info("MODEL ARCHITECTURE")
            self._logger.info("-" * 80)
            self._logger.info("\n%s", model)
            # BUG: FIX THIS
            # total_params, trainable_params = count_parameters(model, dataloader_train)
            # self._logger.info(f"Total parameters:     {total_params:,}")
            # self._logger.info(f"Trainable parameters: {trainable_params:,}")
            self._logger.info("-" * 80)
            if hasattr(model, "initialize_weights"):
                self._logger.info("Model use implemented initialize weight function")
            if hasattr(model, "gradient_clip"):
                self._logger.info("Model use implemented gradient clip function")
            if hasattr(model, "weight_clip"):
                self._logger.info("Model use implemented gradient clip function")
        if hasattr(model, "initialize_weights"):
            model.initialize_weights()
        self.__train_or_continue_train_and_validation_process(
            model=model,
            dataloader_train=dataloader_train,
            dataloader_validation=dataloader_validation,
            batch_size=batch_size,
            img_splits=img_splits,
            epoch_start=0,
            epochs_num=epochs_num,
            loss_function_name=loss_function_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scores_functions_dict=scores_functions_dict,
            early_stop_methods=early_stop_methods,
            early_stop_methods_hyperparameters=early_stop_methods_hyperparameters,
            final_checkpoint_details=final_checkpoint_details,
            min_loss=min_loss,
            loss_list_train_epoch=loss_list_train_epoch,
            loss_list_validation_epoch=loss_list_validation_epoch,
            scores_list_train_epoch=scores_list_train_epoch,
            scores_list_validation_epoch=scores_list_validation_epoch,
            directory_to_save_checkpoint=directory_to_save_checkpoint,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            train_batch_log_threshold=train_batch_log_threshold,
            train_epoch_and_batch_sample_threshold=train_epoch_and_batch_sample_threshold,
            validation_batch_log_threshold=validation_batch_log_threshold,
            validation_epoch_and_batch_sample_threshold=validation_epoch_and_batch_sample_threshold,
        )

    def continue_train(
        self,
        model,
        checkpoint_directory,
        results_directory,
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        train_batch_log_threshold,
        validation_batch_log_threshold,
        new_epochs_num=None,
    ):
        (
            img_splits,
            batch_size,
            checkpoint_epoch,
            epochs_num,
            model_state_dict,
            optimizer_name,
            optimizer_state_dict,
            scheduler_name,
            scheduler_state_dict,
            loss_function_name,
            final_checkpoint_details_dict,
            min_loss,
            loss_list_train_epoch,
            loss_list_validation_epoch,
            scores_list_train_epoch,
            scores_list_validation_epoch,
            augmentation,
        ) = load_checkpoint(checkpoint_directory, results_directory)
        if new_epochs_num is not None:
            epochs_num = new_epochs_num
        model.load_state_dict(model_state_dict)
        model.to(self._device)
        criterion = loss_function_selector(loss_function_name)
        optimizer = optimizer_selector(optimizer_name, model, learning_rate=1e-3)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler = scheduler_selector(scheduler_name, optimizer, None)
        scheduler.load_state_dict(scheduler_state_dict)
        final_checkpoint_details = FinalCheckpointDetails()
        final_checkpoint_details.__dict__.update(final_checkpoint_details_dict)
        scores_name_list = [score_name for score_name in scores_list_train_epoch[0]]
        scores_functions_dict = score_functions_selector(scores_name_list)
        self._logger.info(f"Continue checkpoint {checkpoint_directory}")
        # prepare data
        self.__train_or_continue_train_and_validation_process(
            model=model,
            batch_size=batch_size,
            img_splits=img_splits,
            epoch_start=checkpoint_epoch + 1,
            epochs_num=epochs_num,
            loss_function_name=loss_function_name,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scores_functions_dict=scores_functions_dict,
            final_checkpoint_details=final_checkpoint_details,
            min_loss=min_loss,
            loss_list_train_epoch=loss_list_train_epoch,
            loss_list_validation_epoch=loss_list_validation_epoch,
            scores_list_train_epoch=scores_list_train_epoch,
            scores_list_validation_epoch=scores_list_validation_epoch,
            augmentation=augmentation,
            directory_to_save_checkpoint=directory_to_save_checkpoint,
            directory_to_save_results=directory_to_save_results,
            directory_to_save_samples=directory_to_save_samples,
            train_batch_log_threshold=train_batch_log_threshold,
            validation_batch_log_threshold=validation_batch_log_threshold,
        )

    def __train_or_continue_train_and_validation_process(
        self,
        model,
        dataloader_train,
        dataloader_validation,
        batch_size,
        img_splits,
        epoch_start,
        epochs_num,
        loss_function_name,
        criterion,
        optimizer,
        scheduler,
        scores_functions_dict,
        early_stop_methods,
        early_stop_methods_hyperparameters,
        final_checkpoint_details,
        min_loss,
        loss_list_train_epoch,
        loss_list_validation_epoch,
        scores_list_train_epoch,
        scores_list_validation_epoch,
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        train_batch_log_threshold,
        train_epoch_and_batch_sample_threshold,
        validation_batch_log_threshold,
        validation_epoch_and_batch_sample_threshold,
    ):
        no_improvement_counter = 0
        for epoch_index in range(epoch_start, epochs_num):
            # train
            avg_total_loss_train, epoch_scores_dict_train = self.__train_function(
                model=model,
                dataloader_train=dataloader_train,
                img_splits=img_splits,
                criterion=criterion,
                optimizer=optimizer,
                epoch_index=epoch_index,
                scores_functions_dict=scores_functions_dict,
                directory_to_save_samples=directory_to_save_samples,
                train_batch_log_threshold=train_batch_log_threshold,
                train_epoch_and_batch_sample_threshold=train_epoch_and_batch_sample_threshold,
            )
            loss_list_train_epoch.append(avg_total_loss_train)
            scores_list_train_epoch.append(epoch_scores_dict_train)
            report = generate_report_per_epoch(
                epoch_scores_dict=epoch_scores_dict_train,
                epoch_loss=avg_total_loss_train,
                learning_rate=scheduler.get_last_lr()[0],
                epoch_index=epoch_index,
                epochs_num=epochs_num,
                split=SPLIT_TRAIN_NAME,
            )
            self._logger.info(report)

            # validation
            avg_total_loss_validation, epoch_scores_dict_validation = self.__validation_function(
                model=model,
                dataloader_validation=dataloader_validation,
                criterion=criterion,
                scheduler=scheduler,
                epoch_index=epoch_index,
                scores_functions_dict=scores_functions_dict,
                directory_to_save_samples=directory_to_save_samples,
                validation_batch_log_threshold=validation_batch_log_threshold,
                validation_epoch_and_batch_sample_threshold=validation_epoch_and_batch_sample_threshold,
            )
            loss_list_validation_epoch.append(avg_total_loss_validation)
            scores_list_validation_epoch.append(epoch_scores_dict_validation)
            report = generate_report_per_epoch(
                epoch_scores_dict=epoch_scores_dict_validation,
                epoch_loss=avg_total_loss_validation,
                learning_rate=scheduler.get_last_lr()[0],
                epoch_index=epoch_index,
                epochs_num=epochs_num,
                split=SPLIT_VALIDATION_NAME,
            )
            self._logger.info(report)

            if avg_total_loss_validation < min_loss:
                no_improvement_counter = 0
                min_loss = avg_total_loss_validation
                final_checkpoint_details.set_params(
                    epoch_index,
                    avg_total_loss_train,
                    avg_total_loss_validation,
                    epoch_scores_dict_train,
                    epoch_scores_dict_validation,
                )
                save_checkpoint(
                    batch_size=batch_size,
                    epoch=epoch_index,
                    epochs_num=epochs_num,
                    model_state_dict=model.state_dict(),
                    optimizer_name=type(optimizer).__name__,
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_name=type(scheduler).__name__,
                    scheduler_state_dict=scheduler.state_dict(),
                    loss_function_name=loss_function_name,
                    final_checkpoint_details=final_checkpoint_details,
                    min_loss=min_loss,
                    directory_to_save_checkpoint=directory_to_save_checkpoint,
                )
                save_final_results(
                    loss_list_train_epoch,
                    loss_list_validation_epoch,
                    scores_list_train_epoch,
                    scores_list_validation_epoch,
                    directory_to_save_results,
                )
            else:
                no_improvement_counter += 1
            # early stop methods
            # if (
            #     EARLY_STOP_ACCURACY_LIMIT in early_stop_methods
            #     and epoch_scores_dict_validation[ACCURACY_SCORE_NAME] > early_stop_methods_hyperparameters[ACCURACY_LIMIT]
            # ):
            #     self._logger.info(f"Early stopping triggered: accuracy threshold {early_stop_methods_hyperparameters[ACCURACY_LIMIT]} reached")
            #     break
            if (
                EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT in early_stop_methods
                and no_improvement_counter > early_stop_methods_hyperparameters[LOSS_NO_IMPROVEMENT_EPOCH_LIMIT]
            ):
                self._logger.info(
                    f"Early stopping triggered: no loss improvement for {early_stop_methods_hyperparameters[LOSS_NO_IMPROVEMENT_EPOCH_LIMIT]} epochs"
                )
                break

        # store_channel_importance_counts(
        #     first_20_percent_channels_count_true_prediction=self._first_20_percent_channels_count_train_true_predictions,
        #     second_20_percent_channels_count_true_prediction=self._second_20_percent_channels_count_train_true_predictions,
        #     first_20_percent_channels_count_false_prediction=self._first_20_percent_channels_count_train_false_predictions,
        #     second_20_percent_channels_count_false_prediction=self._second_20_percent_channels_count_train_false_predictions,
        #     total_samples_true_prediction=self._total_train_true_predictions_samples,
        #     total_samples_flase_prediction=self._total_train_false_predictions_samples,
        #     split_name=SPLIT_TRAIN_NAME,
        #     directory_to_save_files=directory_to_save_results,
        # )

        # store_channel_importance_counts(
        #     first_20_percent_channels_count_true_prediction=self._first_20_percent_channels_count_validation_true_predictions,
        #     second_20_percent_channels_count_true_prediction=self._second_20_percent_channels_count_validation_true_predictions,
        #     first_20_percent_channels_count_false_prediction=self._first_20_percent_channels_count_validation_false_predictions,
        #     second_20_percent_channels_count_false_prediction=self._second_20_percent_channels_count_validation_false_predictions,
        #     total_samples_true_prediction=self._total_validation_true_predictions_samples,
        #     total_samples_flase_prediction=self._total_validation_false_predictions_samples,
        #     split_name=SPLIT_VALIDATION_NAME,
        #     directory_to_save_files=directory_to_save_results,
        # )

        save_final_results(loss_list_train_epoch, loss_list_validation_epoch, scores_list_train_epoch, scores_list_validation_epoch, directory_to_save_results)
        report = generate_final_checkpoint_report(final_checkpoint_details)
        self._logger.info(report)

    def __train_function(
        self,
        model,
        dataloader_train,
        img_splits,
        criterion,
        optimizer,
        epoch_index,
        scores_functions_dict,
        directory_to_save_samples,
        train_batch_log_threshold,
        train_epoch_and_batch_sample_threshold,
    ):
        train_batches_num = len(dataloader_train)
        model.train()

        W = next(iter(dataloader_train))[0].shape[-1]
        step = int(W / img_splits)

        vocab_length = len(VOCAB_LETTERS_INDEXES_DICT)
        batch_size = dataloader_train.batch_size

        loss_batch_list = []
        labels_list = []
        outputs_logit_list = []
        for batch_index, (images_batch, labels_batch, labels_lengths_batch) in enumerate(dataloader_train):
            images_batch = images_batch.to(self._device)
            labels_batch_indexed = [letter_to_index_converter(label) for label in labels_batch]
            labels_batch_indexed =  [F.one_hot(torch.tensor(label_indexed, dtype=torch.long), num_classes=vocab_length) for label_indexed in labels_batch_indexed]
            labels_batch_indexed = torch.stack(labels_batch_indexed)
            labels_batch_indexed = labels_batch_indexed.to(self._device)
            hidden = torch.zeros((batch_size, vocab_length), dtype=torch.float32).to(self._device)
            output = get_one_hot_start_token(batch_size, vocab_length).to(self._device)
            # output = get_start_token(batch_size).to(self._device)
            letters = []
            for j in range(img_splits):
                output, hidden = model(images_batch[:, :, :, step * (img_splits - j - 1) : step * (img_splits - j)], output, hidden)
                letters.append(output)

            predicted_word = torch.cat(letters, dim=1)
            # predicted_word = torch.cat(letters, dim=1).view(img_splits, 1, vocab_length)
            # predicted_word = torch.cat(letters, dim=1).view(img_splits, batch_size, vocab_length)

            # if batch_index % 1 == 0:
            #     print(tensor_to_word(predicted_word))
            #     print(indice_to_word(word.view((1, -1))[0]))
            #     print(target_seq_len)
            #     print("=" * 20)

            # compairing the output and word based on CTCLoss
            loss = criterion(predicted_word, labels_batch_indexed, [img_splits] * batch_size, labels_lengths_batch)
            optimizer.zero_grad()
            loss.backward()
            # gradient clip
            if hasattr(model, "gradient_clip"):
                model.gradient_clip()
            # step
            optimizer.step()
            # weight clip
            if hasattr(model, "weight_clip"):
                model.weight_clip()
            # loss
            loss_batch_list.append(loss.item())
            # score
            labels_batch = labels_batch.detach().cpu()

            # output_batch_logit = output_batch_logit.detach().cpu()
            # labels_list.extend(labels_batch)
            # outputs_logit_list.extend(output_batch_logit)
            # predictions_batch = compute_prediction(output_batch_logit)
            # probabilities_batch = compute_probability(output_batch_logit)
            # batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
            #     probabilities_batch, labels_batch, scores_functions_dict
            # )
            # # log
            # if train_batch_log_threshold is not None and batch_index % train_batch_log_threshold == 0:
            #     report = generate_report_per_batch(
            #         batch_scores_dict=batch_scores_dict,
            #         batch_loss=loss.item(),
            #         batch_index=batch_index,
            #         batches_num=train_batches_num,
            #         split=SPLIT_TRAIN_NAME,
            #     )
            #     self._logger.info(report)
            # if (
            #     train_epoch_and_batch_sample_threshold is not None
            #     and epoch_index % train_epoch_and_batch_sample_threshold[0] == 0
            #     and batch_index % train_epoch_and_batch_sample_threshold[1] == 0
            # ):
            #     batch_size = images_batch.shape[0]
            #     random_index = torch.randint(0, batch_size, (1,)).item()
            #     sample = images_batch[random_index].detach()
            #     sample_true_label = labels_batch[random_index].detach()
            #     sample_predicted_label = predictions_batch[random_index]
            #     channel_importance = compute_integrated_gradients(
            #         model=model,
            #         x=sample,
            #         target_class=1,
            #     )
            #     channel_importance = channel_importance.detach().cpu()
            #     if channel_importance.ndim == 2:
            #         channel_importance = concatenate_hemispharer_based_separated_channels(
            #             channel_importance[0],
            #             channel_importance[1],
            #         )

            #     ch_names = dataloader_train.dataset.get_selected_channels()
            #     channel_importance = channel_importance.numpy()
            #     n_channels = len(channel_importance)
            #     sorted_channel_indices = np.argsort(channel_importance)[::-1]
            #     first_20_percent_count = int(np.ceil(0.20 * n_channels))
            #     second_20_percent_count = int(np.ceil(0.20 * n_channels))
            #     first_20_percent_indices = sorted_channel_indices[:first_20_percent_count]
            #     second_20_percent_indices = sorted_channel_indices[first_20_percent_count : first_20_percent_count + second_20_percent_count]

            #     is_correct_prediction = sample_predicted_label == sample_true_label
            #     if is_correct_prediction:
            #         first_20_percent_counter = self._first_20_percent_channels_count_train_true_predictions
            #         second_20_percent_counter = self._second_20_percent_channels_count_train_true_predictions
            #         self._total_train_true_predictions_samples += 1
            #     else:
            #         first_20_percent_counter = self._first_20_percent_channels_count_train_false_predictions
            #         second_20_percent_counter = self._second_20_percent_channels_count_train_false_predictions
            #         self._total_train_false_predictions_samples += 1

            #     for channel_index in first_20_percent_indices:
            #         first_20_percent_counter[ch_names[channel_index]] += 1
            #     for channel_index in second_20_percent_indices:
            #         second_20_percent_counter[ch_names[channel_index]] += 1

            #     VisualizerParent.visualize_channels_importance_on_2d_head(
            #         channel_scores=channel_importance,
            #         split=SPLIT_TRAIN_NAME,
            #         file_name=self._train_sample_index,
            #         montage_name=dataloader_train.dataset.get_montage_name(),
            #         ch_names=dataloader_train.dataset.get_selected_channels(),
            #         true_label=sample_true_label,
            #         predicted_label=sample_predicted_label,
            #         directory_to_save_figure=directory_to_save_samples,
            #     )
            #     self._train_sample_index += 1
        avg_total_loss_train = sum(loss_batch_list) / train_batches_num
        # outputs_logit_list = torch.stack(outputs_logit_list)
        # labels_list = torch.stack(labels_list)
        # predictions_epoch = compute_prediction(outputs_logit_list)
        # probabilities_epoch = compute_probability(outputs_logit_list)
        # epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
        #     probabilities_epoch, labels_list, scores_functions_dict
        # )
        return avg_total_loss_train, None
        # return avg_total_loss_train, epoch_scores_dict

    def __validation_function(
        self,
        model,
        dataloader_validation,
        img_splits,
        criterion,
        scheduler,
        epoch_index,
        scores_functions_dict,
        directory_to_save_samples,
        validation_batch_log_threshold,
        validation_epoch_and_batch_sample_threshold,
    ):
        validation_batches_num = len(dataloader_validation)
        with torch.no_grad():
            model.eval()
            W = next(iter(dataloader_validation))[0].shape[-1]
            step = int(W / img_splits)
            vocab_length = len(VOCAB_LETTERS_INDEXES_DICT)
            batch_size = dataloader_validation.batch_size

            loss_batch_list = []
            labels_list = []
            outputs_logit_list = []

            for batch_index, (images_batch, labels_batch, labels_lengths_batch) in enumerate(dataloader_validation):
                images_batch, labels_batch = images_batch.to(self._device), labels_batch.to(self._device)
                hidden = torch.zeros((1, vocab_length)).double().to(self._device)
                # output = get_one_hot_start_token(batch_size, vocab_length).to(self._device)
                output = get_start_token(batch_size).to(self._device)
                letters = []
                for j in range(img_splits):
                    output, hidden = model(images_batch[:, :, :, step * (img_splits - j - 1) : step * (img_splits - j)], output, hidden)
                    letters.append(output)

                predicted_word = torch.cat(letters, dim=1)
                # predicted_word = torch.cat(letters, dim=1).view(img_splits, 1, vocab_length)
                # predicted_word = torch.cat(letters, dim=1).view(img_splits, batch_size, vocab_length)

                # if batch_index % 1 == 0:
                #     print(tensor_to_word(predicted_word))
                #     print(indice_to_word(word.view((1, -1))[0]))
                #     print(target_seq_len)
                #     print("=" * 20)

                # compairing the output and word based on CTCLoss
                loss = criterion(predicted_word, labels_batch, [img_splits] * batch_size, labels_lengths_batch)
                # loss
                loss_batch_list.append(loss.item())

                # # score
                # labels_batch = labels_batch.cpu()
                # output_batch_logit = output_batch_logit.cpu()
                # labels_list.extend(labels_batch)
                # outputs_logit_list.extend(output_batch_logit)

                # predictions_batch = compute_prediction(output_batch_logit)
                # probabilities_batch = compute_probability(output_batch_logit)
                # batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
                #     probabilities_batch, labels_batch, scores_functions_dict
                # )
                # # log
                # if validation_batch_log_threshold is not None and batch_index % validation_batch_log_threshold == 0:
                #     report = generate_report_per_batch(
                #         batch_scores_dict=batch_scores_dict,
                #         batch_loss=loss.item(),
                #         batch_index=batch_index,
                #         batches_num=validation_batches_num,
                #         split=SPLIT_VALIDATION_NAME,
                #     )
                #     self._logger.info(report)
                # if (
                #     validation_epoch_and_batch_sample_threshold is not None
                #     and epoch_index % validation_epoch_and_batch_sample_threshold[0] == 0
                #     and batch_index % validation_epoch_and_batch_sample_threshold[1] == 0
                # ):
                #     batch_size = images_batch.shape[0]
                #     random_index = torch.randint(0, batch_size, (1,)).item()
                #     sample = images_batch[random_index].detach()
                #     sample_true_label = labels_batch[random_index].detach()
                #     sample_predicted_label = predictions_batch[random_index]
                #     channel_importance = compute_integrated_gradients(
                #         model=model,
                #         x=sample,
                #         target_class=1,
                #     )
                #     channel_importance = channel_importance.detach().cpu()
                #     if channel_importance.ndim == 2:
                #         channel_importance = concatenate_hemispharer_based_separated_channels(
                #             channel_importance[0],
                #             channel_importance[1],
                #         )

                #     ch_names = dataloader_validation.dataset.get_selected_channels()
                #     channel_importance = channel_importance.numpy()
                #     n_channels = len(channel_importance)
                #     sorted_channel_indices = np.argsort(channel_importance)[::-1]
                #     first_20_percent_count = int(np.ceil(0.20 * n_channels))
                #     second_20_percent_count = int(np.ceil(0.20 * n_channels))
                #     first_20_percent_indices = sorted_channel_indices[:first_20_percent_count]
                #     second_20_percent_indices = sorted_channel_indices[first_20_percent_count : first_20_percent_count + second_20_percent_count]

                #     is_correct_prediction = sample_predicted_label == sample_true_label
                #     if is_correct_prediction:
                #         first_20_percent_counter = self._first_20_percent_channels_count_validation_true_predictions
                #         second_20_percent_counter = self._second_20_percent_channels_count_validation_true_predictions
                #         self._total_validation_true_predictions_samples += 1
                #     else:
                #         first_20_percent_counter = self._first_20_percent_channels_count_validation_false_predictions
                #         second_20_percent_counter = self._second_20_percent_channels_count_validation_false_predictions
                #         self._total_validation_false_predictions_samples += 1

                #     for channel_index in first_20_percent_indices:
                #         first_20_percent_counter[ch_names[channel_index]] += 1
                #     for channel_index in second_20_percent_indices:
                #         second_20_percent_counter[ch_names[channel_index]] += 1

                #     VisualizerParent.visualize_channels_importance_on_2d_head(
                #         channel_scores=channel_importance,
                #         split=SPLIT_VALIDATION_NAME,
                #         file_name=self._validation_sample_index,
                #         montage_name=dataloader_validation.dataset.get_montage_name(),
                #         ch_names=dataloader_validation.dataset.get_selected_channels(),
                #         true_label=sample_true_label,
                #         predicted_label=sample_predicted_label,
                #         directory_to_save_figure=directory_to_save_samples,
                #     )
                #     self._validation_sample_index += 1
            avg_total_loss_validation = sum(loss_batch_list) / validation_batches_num
            # outputs_logit_list = torch.stack(outputs_logit_list)
            # labels_list = torch.stack(labels_list)
            # predictions_epoch = compute_prediction(outputs_logit_list)
            # probabilities_epoch = compute_probability(outputs_logit_list)
            # epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
            #     probabilities_epoch, labels_list, scores_functions_dict
            # )
            if type(scheduler).__name__ == SCHEDULER_PLATEAU_NAME:
                scheduler.step(avg_total_loss_validation)
            else:
                scheduler.step()
            return avg_total_loss_validation, None
            # return avg_total_loss_validation, epoch_scores_dict
