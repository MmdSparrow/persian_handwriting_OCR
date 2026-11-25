import numpy as np
import torch
from torch.utils.data import DataLoader

# from visualizer import VisualizerParent
from constants.common_constants import (
    CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
    CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY,
    SCHEDULER_PLATEAU_NAME,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)
from data import DatasetParent
from utils.alphabet_letters_utils import clean_encoded_labels, index_to_letter_converter, letter_to_index_converter
from utils.experiment_utils.experiment_common_utils import (
    compute_scores,
    compute_scores_dict_list_avg,
    score_functions_selector,
)
from utils.experiment_utils.gradient_descent_utils.checkpoint_utils import load_checkpoint, save_checkpoint
from utils.experiment_utils.gradient_descent_utils.experiment_gradient_descent_common_utils import (
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
from utils.load_and_store_dataset_preprocess_info import store_dataset_preprocesses_info

from ._train_parent import TrainParent

__all__ = ["TrainGradientDescentParent"]


class TrainGradientDescentParent(TrainParent):
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
            total_params, trainable_params = count_parameters(model)
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
        self.__train_or_continue_train_and_validation_process(
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
        batches_num = len(dataloader_train)
        model.train()

        batch_size = dataloader_train.batch_size

        loss_batch_list = []
        batch_scores_dict_list = []
        for batch_index, (images_batch, labels_batch, labels_lengths_batch) in enumerate(dataloader_train):
            images_batch = images_batch.to(self._device)
            labels_batch_encoded = [letter_to_index_converter(label) for label in labels_batch]
            labels_batch_encoded = torch.stack(labels_batch_encoded)
            labels_batch_encoded = labels_batch_encoded.to(self._device)

            decoder_inputs = labels_batch_encoded[:, :-1]
            targets = labels_batch_encoded[:, 1:]

            features, hidden, padding_mask = model.encode(images_batch)

            # decode
            outputs = []
            sequence_length = decoder_inputs.size(1)
            input_token = decoder_inputs[:, 0]

            for t in range(sequence_length):
                logits, hidden, _ = model.decode_step(
                    features=features,
                    input_token=input_token,
                    hidden=hidden,
                    padding_mask=padding_mask,
                )
                outputs.append(logits)

                if t + 1 < sequence_length:
                    input_token = decoder_inputs[:, t + 1]

            outputs = torch.stack(outputs, dim=1)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )
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
            predictions_encoded = outputs.argmax(dim=2).detach().cpu()
            predictions_encoded = clean_encoded_labels(predictions_encoded)
            ground_truth = clean_encoded_labels(labels_batch_encoded.detach().cpu())
            batch_scores_dict = compute_scores(
                predictions=predictions_encoded,
                ground_truth=ground_truth,
                scores_functions_dict=scores_functions_dict,
            )
            batch_scores_dict_list.append(batch_scores_dict)

            # log
            if train_batch_log_threshold is not None and batch_index % train_batch_log_threshold == 0:
                report = generate_report_per_batch(
                    batch_scores_dict=batch_scores_dict,
                    batch_loss=loss.item(),
                    batch_index=batch_index,
                    batches_num=batches_num,
                    split=SPLIT_TRAIN_NAME,
                )
                self._logger.info(report)
            if (
                train_epoch_and_batch_sample_threshold is not None
                and epoch_index % train_epoch_and_batch_sample_threshold[0] == 0
                and batch_index % train_epoch_and_batch_sample_threshold[1] == 0
            ):
                random_index = torch.randint(0, batch_size, (1,)).item()
                sample_ground_truth = ground_truth[random_index]
                sample_predicted_label = predictions_encoded[random_index]

                print("-" * 10)
                print(f"Train sample index: {self._train_sample_index}")
                print(f"Ground Truth: {index_to_letter_converter(sample_ground_truth)}")
                print(f"Prediction :{index_to_letter_converter(sample_predicted_label)}")
                print("-" * 10)
                self._train_sample_index += 1

        avg_total_loss_train = sum(loss_batch_list) / batches_num
        epoch_scores_dict = compute_scores_dict_list_avg(batch_scores_dict_list)
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
        batches_num = len(dataloader_validation)
        model.eval()

        batch_size = dataloader_validation.batch_size

        loss_batch_list = []
        batch_scores_dict_list = []
        for batch_index, (images_batch, labels_batch, labels_lengths_batch) in enumerate(dataloader_validation):
            images_batch = images_batch.to(self._device)
            labels_batch_encoded = [letter_to_index_converter(label) for label in labels_batch]
            labels_batch_encoded = torch.stack(labels_batch_encoded)
            labels_batch_encoded = labels_batch_encoded.to(self._device)

            decoder_inputs = labels_batch_encoded[:, :-1]
            targets = labels_batch_encoded[:, 1:]

            features, hidden, padding_mask = model.encode(images_batch)

            # decode
            outputs = []
            sequence_length = decoder_inputs.size(1)
            input_token = decoder_inputs[:, 0]

            for t in range(sequence_length):
                logits, hidden, _ = model.decode_step(
                    features=features,
                    input_token=input_token,
                    hidden=hidden,
                    padding_mask=padding_mask,
                )
                outputs.append(logits)

                if t + 1 < sequence_length:
                    input_token = logits.argmax(dim=1)

            outputs = torch.stack(outputs, dim=1)
            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
            )
            loss_batch_list.append(loss.item())
            # score
            labels_batch = labels_batch.detach().cpu()

            # compute scores
            predictions_encoded = outputs.argmax(dim=1).detach().cpu()
            predictions_encoded = clean_encoded_labels(predictions_encoded)
            ground_truth = clean_encoded_labels(labels_batch_encoded.detach().cpu())
            batch_scores_dict = compute_scores(
                predictions=predictions_encoded,
                ground_truth=ground_truth,
                scores_functions_dict=scores_functions_dict,
            )
            batch_scores_dict_list.append(batch_scores_dict)

            # log
            if validation_batch_log_threshold is not None and batch_index % validation_batch_log_threshold == 0:
                report = generate_report_per_batch(
                    batch_scores_dict=batch_scores_dict,
                    batch_loss=loss.item(),
                    batch_index=batch_index,
                    batches_num=batches_num,
                    split=SPLIT_VALIDATION_NAME,
                )
                self._logger.info(report)
            if (
                validation_epoch_and_batch_sample_threshold is not None
                and epoch_index % validation_epoch_and_batch_sample_threshold[0] == 0
                and batch_index % validation_epoch_and_batch_sample_threshold[1] == 0
            ):
                random_index = torch.randint(0, batch_size, (1,)).item()
                sample_ground_truth = ground_truth[random_index].detach()
                sample_predicted_label = predictions_encoded[random_index]

                print("-" * 10)
                print(f"Validation sample index: {self._validation_sample_index}")
                print(f"Ground Truth: {index_to_letter_converter(sample_ground_truth)}")
                print(f"Prediction :{index_to_letter_converter(sample_predicted_label)}")
                print("-" * 10)
                self._validation_sample_index += 1

        avg_total_loss_validation = sum(loss_batch_list) / batches_num
        epoch_scores_dict = compute_scores_dict_list_avg(batch_scores_dict_list)
        if type(scheduler).__name__ == SCHEDULER_PLATEAU_NAME:
            scheduler.step(avg_total_loss_validation)
        else:
            scheduler.step()
        return avg_total_loss_validation, epoch_scores_dict
