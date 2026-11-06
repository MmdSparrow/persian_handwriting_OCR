import numpy as np
import torch
from torch.utils.data import DataLoader

from constants.common_constants import (
    ACCURACY_LIMIT,
    ACCURACY_SCORE_NAME,
    EARLY_STOP_ACCURACY_LIMIT,
    EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    SCHEDULER_PLATEAU_NAME,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)
from data import DatasetParent
from info.common_info import PERSIAN_ALPHABETS_CHARACTERS, SPECIAL_CHARACTERS
from utils.experiment_utils.experiment_common_utils import (
    compute_prediction_based_scores,
    compute_probability_based_scores,
    score_functions_selector,
    store_channel_importance_counts,
)
from utils.experiment_utils.gradient_descent_utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.experiment_utils.gradient_descent_utils.experiment_gradient_descent_common_utils import (
    compute_prediction,
    compute_probability,
    count_parameters,
    generate_report_per_batch,
    loss_function_selector,
)
from utils.experiment_utils.gradient_descent_utils.train_gradient_descent_parent_utils import (
    FinalCheckpointDetails,
    generate_final_checkpoint_report,
    generate_report_per_epoch,
    optimizer_selector,
    save_final_results,
    scheduler_selector,
)

from ._train_parent import TrainParent

__all__ = ["TrainGradientDescentParent"]


class TrainGradientDescentParent(TrainParent):
    def __init__(self, dataset_directory, image_splits_num, word_max_length, logger, log_init_info):
        super().__init__(dataset_directory, logger, log_init_info)
        if log_init_info:
            self._logger.info("If model has `gradient_clip`, `weight_clip` or `initialize_weights` methods, they are called automatically")

        self.__training_vocabulary = PERSIAN_ALPHABETS_CHARACTERS + SPECIAL_CHARACTERS

        self.__letter_to_index_dict = {}
        
        self.__encode_dimension= len(self.__training_vocabulary)
        for i in range(self.__encode_dimension):
            self.__letter_to_index_dict[self.__training_vocabulary[i]] = i

        self.__image_splits_num = image_splits_num
        self.__word_max_length = word_max_length

    def __word_preprocess(self, word):
        if len(word) > self.__word_max_length:
            word = word[: self.__word_max_length]
        word = word + "E"
        return [self.__letter_to_index_dict[c] for c in word]

    def __prepare_data(self, batch_size, transform=None, augmentation=None):
        dataset_train = DatasetParent(
            root=self.dataset_directory,
            split=SPLIT_TRAIN_NAME,
            transform=transform,
            augmentation=augmentation,
        )
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )
        dataset_validation = DatasetParent(
            root=self.dataset_directory,
            split=SPLIT_VALIDATION_NAME,
            transform=transform,
            augmentation=None,
        )
        dataloader_validation = DataLoader(
            dataset_validation,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader_train, dataloader_validation

    def train_and_validation(
        self,
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
        directory_to_save_checkpoint,
        directory_to_save_results,
        directory_to_save_samples,
        train_batch_log_threshold,
        train_epoch_and_batch_epoch_sample_threshold,
        train_epoch_and_batch_batch_sample_threshold,
        validation_batch_log_threshold,
        validation_epoch_and_batch_epoch_sample_threshold,
        validation_epoch_and_batch_batch_sample_threshold,
        log_init_info,
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
        dataloader_train, dataloader_validation = self.__prepare_data(batch_size, None, augmentation)
        if log_init_info:
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
            total_params, trainable_params = count_parameters(model, dataloader_train)
            self._logger.info(f"Total parameters:     {total_params:,}")
            self._logger.info(f"Trainable parameters: {trainable_params:,}")
            self._logger.info("-" * 80)
            if hasattr(model, "initialize_weights"):
                self._logger.info("Model use implemented initialize weight function")
            if hasattr(model, "gradient_clip"):
                self._logger.info("Model use implemented gradient clip function")
            if hasattr(model, "weight_clip"):
                self._logger.info("Model use implemented gradient clip function")
        if hasattr(model, "initialize_weights"):
            model.initialize_weights()

        self.__train_or_continue_train_and_validation(
            model=model,
            dataloader_train=dataloader_train,
            dataloader_validation=dataloader_validation,
            batch_size=batch_size,
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

    def __train_or_continue_train_and_validation(
        self,
        model,
        dataloader_train,
        dataloader_validation,
        validation_function,
        batch_size,
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
            avg_total_loss_validation, epoch_scores_dict_validation = validation_function(
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
            if (
                EARLY_STOP_ACCURACY_LIMIT in early_stop_methods
                and epoch_scores_dict_validation[ACCURACY_SCORE_NAME] > early_stop_methods_hyperparameters[ACCURACY_LIMIT]
            ):
                self._logger.info(f"Early stopping triggered: accuracy threshold {early_stop_methods_hyperparameters[ACCURACY_LIMIT]} reached")
                break
            if (
                EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT in early_stop_methods
                and no_improvement_counter > early_stop_methods_hyperparameters[LOSS_NO_IMPROVEMENT_EPOCH_LIMIT]
            ):
                self._logger.info(
                    f"Early stopping triggered: no loss improvement for {early_stop_methods_hyperparameters[LOSS_NO_IMPROVEMENT_EPOCH_LIMIT]} epochs"
                )
                break

        store_channel_importance_counts(
            first_20_percent_channels_count_true_prediction=self._first_20_percent_channels_count_train_true_predictions,
            second_20_percent_channels_count_true_prediction=self._second_20_percent_channels_count_train_true_predictions,
            first_20_percent_channels_count_false_prediction=self._first_20_percent_channels_count_train_false_predictions,
            second_20_percent_channels_count_false_prediction=self._second_20_percent_channels_count_train_false_predictions,
            total_samples_true_prediction=self._total_train_true_predictions_samples,
            total_samples_flase_prediction=self._total_train_false_predictions_samples,
            split_name=SPLIT_TRAIN_NAME,
            directory_to_save_files=directory_to_save_results,
        )

        store_channel_importance_counts(
            first_20_percent_channels_count_true_prediction=self._first_20_percent_channels_count_validation_true_predictions,
            second_20_percent_channels_count_true_prediction=self._second_20_percent_channels_count_validation_true_predictions,
            first_20_percent_channels_count_false_prediction=self._first_20_percent_channels_count_validation_false_predictions,
            second_20_percent_channels_count_false_prediction=self._second_20_percent_channels_count_validation_false_predictions,
            total_samples_true_prediction=self._total_validation_true_predictions_samples,
            total_samples_flase_prediction=self._total_validation_false_predictions_samples,
            split_name=SPLIT_VALIDATION_NAME,
            directory_to_save_files=directory_to_save_results,
        )

        save_final_results(loss_list_train_epoch, loss_list_validation_epoch, scores_list_train_epoch, scores_list_validation_epoch, directory_to_save_results)
        report = generate_final_checkpoint_report(final_checkpoint_details)
        self._logger.info(report)

    def __train_function(
        self,
        model,
        dataloader_train,
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
        loss_batch_list = []
        labels_list = []
        outputs_logit_list = []
        for batch_index, (images_batch, labels_batch) in enumerate(dataloader_train):
            images_batch, labels_batch = images_batch.to(self._device), labels_batch.to(self._device)

            # preprocess labels
            self.__word_preprocess
            def __word_preprocess(self, word):
                if len(word) > self.__word_max_length:
                    word = word[: self.__word_max_length]
                word = word + "E"
                return [self.__letter_to_index_dict[c] for c in word]

            
            hidden = torch.zeros((1, self.__encode_dimension)).double().to(self._device)

            start_token = torch.zeros((1, self.__encode_dimension), dtype=torch.double).to(self._device)
            start_token[0][self.__letter_to_index_dict["S"]] = 1

            _, _, image_width = images_batch.shape
            step = int(image_width / self.__image_splits_num)

            output = start_token
            letters = []
            for j in range(self.__image_splits_num):
                output, hidden = model(images_batch[:, :, :, step * (self.__image_splits_num - j - 1) : step * (self.__image_splits_num - j)], output, hidden)
                letters.append(output)

            predicted_word = torch.cat(letters, dim=1).view(self.__image_splits_num, 1, self.__encode_dimension)
            # target_seq_len = torch.tensor([word.size(1)], dtype=torch.long)
            
            
            output_batch_logit = model(images_batch)
            loss = criterion(output_batch_logit, labels_batch)
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
            output_batch_logit = output_batch_logit.detach().cpu()
            labels_list.extend(labels_batch)
            outputs_logit_list.extend(output_batch_logit)

            predictions_batch = compute_prediction(output_batch_logit)
            probabilities_batch = compute_probability(output_batch_logit)
            batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
                probabilities_batch, labels_batch, scores_functions_dict
            )
            # log
            if train_batch_log_threshold is not None and batch_index % train_batch_log_threshold == 0:
                report = generate_report_per_batch(
                    batch_scores_dict=batch_scores_dict,
                    batch_loss=loss.item(),
                    batch_index=batch_index,
                    batches_num=train_batches_num,
                    split=SPLIT_TRAIN_NAME,
                )
                self._logger.info(report)
            if (
                train_epoch_and_batch_sample_threshold is not None
                and epoch_index % train_epoch_and_batch_sample_threshold[0] == 0
                and batch_index % train_epoch_and_batch_sample_threshold[1] == 0
            ):
                batch_size = images_batch.shape[0]
                random_index = torch.randint(0, batch_size, (1,)).item()
                sample = images_batch[random_index].detach()
                sample_true_label = labels_batch[random_index].detach()
                sample_predicted_label = predictions_batch[random_index]
                ch_names = dataloader_train.dataset.get_selected_channels()

                is_correct_prediction = sample_predicted_label == sample_true_label
                if is_correct_prediction:
                    first_20_percent_counter = self._first_20_percent_channels_count_train_true_predictions
                    second_20_percent_counter = self._second_20_percent_channels_count_train_true_predictions
                    self._total_train_true_predictions_samples += 1
                else:
                    first_20_percent_counter = self._first_20_percent_channels_count_train_false_predictions
                    second_20_percent_counter = self._second_20_percent_channels_count_train_false_predictions
                    self._total_train_false_predictions_samples += 1

                # VisualizerParent.visualize_channels_importance_on_2d_head(
                #     channel_scores=channel_importance,
                #     split=SPLIT_TRAIN_NAME,
                #     file_name=self._train_sample_index,
                #     montage_name=dataloader_train.dataset.get_montage_name(),
                #     ch_names=dataloader_train.dataset.get_selected_channels(),
                #     true_label=sample_true_label,
                #     predicted_label=sample_predicted_label,
                #     directory_to_save_figure=directory_to_save_samples,
                # )
                self._train_sample_index += 1
        avg_total_loss_train = sum(loss_batch_list) / train_batches_num
        outputs_logit_list = torch.stack(outputs_logit_list)
        labels_list = torch.stack(labels_list)
        predictions_epoch = compute_prediction(outputs_logit_list)
        probabilities_epoch = compute_probability(outputs_logit_list)
        epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
            probabilities_epoch, labels_list, scores_functions_dict
        )
        return avg_total_loss_train, epoch_scores_dict

    def __validation_function(
        self,
        model,
        dataloader_validation,
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
            loss_batch_list = []
            labels_list = []
            outputs_logit_list = []
            for batch_index, (eegs_batch, labels_batch) in enumerate(dataloader_validation):
                eegs_batch, labels_batch = eegs_batch.to(self._device), labels_batch.to(self._device)
                output_batch_logit = model(eegs_batch)
                loss = criterion(output_batch_logit, labels_batch)
                # loss
                loss_batch_list.append(loss.item())
                # score
                labels_batch = labels_batch.cpu()
                output_batch_logit = output_batch_logit.cpu()
                labels_list.extend(labels_batch)
                outputs_logit_list.extend(output_batch_logit)

                predictions_batch = compute_prediction(output_batch_logit)
                probabilities_batch = compute_probability(output_batch_logit)
                batch_scores_dict = compute_prediction_based_scores(predictions_batch, labels_batch, scores_functions_dict) | compute_probability_based_scores(
                    probabilities_batch, labels_batch, scores_functions_dict
                )
                # log
                if validation_batch_log_threshold is not None and batch_index % validation_batch_log_threshold == 0:
                    report = generate_report_per_batch(
                        batch_scores_dict=batch_scores_dict,
                        batch_loss=loss.item(),
                        batch_index=batch_index,
                        batches_num=validation_batches_num,
                        split=SPLIT_VALIDATION_NAME,
                    )
                    self._logger.info(report)
                if (
                    validation_epoch_and_batch_sample_threshold is not None
                    and epoch_index % validation_epoch_and_batch_sample_threshold[0] == 0
                    and batch_index % validation_epoch_and_batch_sample_threshold[1] == 0
                ):
                    batch_size = eegs_batch.shape[0]
                    random_index = torch.randint(0, batch_size, (1,)).item()
                    sample = eegs_batch[random_index].detach()
                    sample_true_label = labels_batch[random_index].detach()
                    sample_predicted_label = predictions_batch[random_index]

                    ch_names = dataloader_validation.dataset.get_selected_channels()

                    is_correct_prediction = sample_predicted_label == sample_true_label
                    if is_correct_prediction:
                        self._total_validation_true_predictions_samples += 1
                    else:
                        self._total_validation_false_predictions_samples += 1

                    # VisualizerParent.visualize_channels_importance_on_2d_head(
                    #     channel_scores=channel_importance,
                    #     split=SPLIT_VALIDATION_NAME,
                    #     file_name=self._validation_sample_index,
                    #     montage_name=dataloader_validation.dataset.get_montage_name(),
                    #     ch_names=dataloader_validation.dataset.get_selected_channels(),
                    #     true_label=sample_true_label,
                    #     predicted_label=sample_predicted_label,
                    #     directory_to_save_figure=directory_to_save_samples,
                    # )
                    self._validation_sample_index += 1
            avg_total_loss_validation = sum(loss_batch_list) / validation_batches_num
            outputs_logit_list = torch.stack(outputs_logit_list)
            labels_list = torch.stack(labels_list)
            predictions_epoch = compute_prediction(outputs_logit_list)
            probabilities_epoch = compute_probability(outputs_logit_list)
            epoch_scores_dict = compute_prediction_based_scores(predictions_epoch, labels_list, scores_functions_dict) | compute_probability_based_scores(
                probabilities_epoch, labels_list, scores_functions_dict
            )
            if type(scheduler).__name__ == SCHEDULER_PLATEAU_NAME:
                scheduler.step(avg_total_loss_validation)
            else:
                scheduler.step()
            return avg_total_loss_validation, epoch_scores_dict
