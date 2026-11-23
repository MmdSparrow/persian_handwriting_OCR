import torch


class BaseExperiment:
    def __select_device(self, log_init_info):
        if torch.cuda.is_available():
            if log_init_info:
                self._logger.info("-" * 80)
                self._logger.info("CUDA is available")
                self._logger.info(f"GPU count: {torch.cuda.device_count()}")
                self._logger.info(f"Current device: {torch.cuda.current_device()}")
                self._logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
                self._logger.info("-" * 80)
            self._device = torch.device("cuda")
        else:
            if log_init_info:
                self._logger.info("-" * 80)
                self._logger.info("CUDA is not available")
                self._logger.info("-" * 80)
            self._device = torch.device("cpu")

    def __init__(self, dataset_directory, outputs_directory, logger, seed, log_init_info):
        self._dataset_directory = dataset_directory
        self._outputs_directory = outputs_directory
        self._logger = logger
        self._seed = seed
        self.__select_device(log_init_info)
        self._log_init_info = log_init_info

    def _log_chosen_preprocesses_info(
        self,
        chosen_preprocesses_name_list,
        chosen_preprocesses_hyperparameters_dict,
        preprocesses_name_list_in_order_of_apply,
        preprocesses_and_their_hyperparameters_dict,
        preprocesses_type,
    ):
        chosen_preprocesses_name_list.sort(key=preprocesses_name_list_in_order_of_apply.index)
        self._logger.info(f"Order of applying {preprocesses_type} preprocesses:")
        for i, preprocess_name in enumerate(preprocesses_name_list_in_order_of_apply):
            self._logger.info(f"{i + 1}- {preprocess_name}")
        self._logger.info(f"Chosen {preprocesses_type} preprocesses in order: {chosen_preprocesses_name_list}")
        if chosen_preprocesses_name_list:
            self._logger.info(f"{preprocesses_type.capitalize()} preprocesses hyperparameters:")
            for preprocess_name in chosen_preprocesses_name_list:
                if preprocess_name in preprocesses_and_their_hyperparameters_dict:
                    for hyperparameter_name in preprocesses_and_their_hyperparameters_dict[preprocess_name]:
                        self._logger.info(
                            f"{hyperparameter_name.replace('_', ' ').capitalize()}: {chosen_preprocesses_hyperparameters_dict[hyperparameter_name]}"
                        )
