import torch
from torch.utils.data import DataLoader

from constants.common_constants import SPLIT_TEST_NAME
from data import DatasetParent
from utils.alphabet_letters_utils import clean_encoded_labels, index_to_letter_converter, letter_to_index_converter
from utils.experiment_utils.experiment_common_utils import (
    compute_scores,
    compute_scores_dict_list_avg,
    score_functions_selector,
)
from utils.experiment_utils.gradient_descent_utils.checkpoint_utils import load_final_checkpoint
from utils.experiment_utils.gradient_descent_utils.experiment_gradient_descent_common_utils import (
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
            total_params, trainable_params = count_parameters(model)
            self._logger.info(f"Total parameters:     {total_params:,}")
            self._logger.info(f"Trainable parameters: {trainable_params:,}")
            self._logger.info("-" * 80)

        self.__test_function(
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
        batches_num = len(dataloader_test)
        model.eval()

        batch_size = dataloader_test.batch_size

        loss_batch_list = []
        batch_scores_dict_list = []
        for batch_index, (images_batch, labels_batch, labels_lengths_batch) in enumerate(dataloader_test):
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
            if test_batch_log_threshold is not None and batch_index % test_batch_log_threshold == 0:
                report = generate_report_per_batch(
                    batch_scores_dict=batch_scores_dict,
                    batch_loss=loss.item(),
                    batch_index=batch_index,
                    batches_num=batches_num,
                    split=SPLIT_TEST_NAME,
                )
                self._logger.info(report)
            if test_batch_sample_threshold is not None and batch_index % test_batch_sample_threshold == 0:
                random_index = torch.randint(0, batch_size, (1,)).item()
                sample_ground_truth = ground_truth[random_index].detach()
                sample_predicted_label = predictions_encoded[random_index]

                print("-" * 10)
                print(f"Test sample index: {self._test_sample_index}")
                print(f"Ground Truth: {index_to_letter_converter(sample_ground_truth)}")
                print(f"Prediction :{index_to_letter_converter(sample_predicted_label)}")
                print("-" * 10)
                self._test_sample_index += 1

        avg_total_loss_test = sum(loss_batch_list) / batches_num
        epoch_scores_dict = compute_scores_dict_list_avg(batch_scores_dict_list)
        save_final_results(avg_total_loss_test, epoch_scores_dict, directory_to_save_results)
        report = generate_final_report(epoch_scores_dict, avg_total_loss_test)
        self._logger.info(report)
        return avg_total_loss_test, epoch_scores_dict
