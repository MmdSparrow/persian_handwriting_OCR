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

    def __init__(self, dataset_root, logger, log_init_info):
        self.dataset_root = dataset_root
        self._logger = logger
        self.__select_device(log_init_info)
