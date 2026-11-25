from configs.common_configs import MY_LUCKY_NUMBER
from constants.base_directories_names_constants import DATASETS_ROOT_DIRECTORY_NAME
from constants.common_constants import (
    ACCURACY_LIMIT,
    ALL_SCORES_NAMES_LIST,
    CROSS_ENTROPY_LOSS,
    EARLY_STOP_ACCURACY_LIMIT,
    EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
    OPTIMIZER_ADAM_NAME,
    SCHEDULER_PLATEAU_NAME,
    TYPED_WORDS_IMAGES_DATASET_NAME,
)
from info.vocab_info import PADDING_CHARACTER, VOCAB_LETTERS_INDEXES_DICT
from interfaces import DataFunctionsInterface, NeuralNetworkFunctionsInterface
from models import CNNAttentionOCR
from runners.setup import set_seed
from utils.dataset_downloader import donwload_dataset
from utils.load_and_store_folds_and_split_info import get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split

set_seed(MY_LUCKY_NUMBER)
############################## CONFIGS ##############################
SELECTED_DATASET = TYPED_WORDS_IMAGES_DATASET_NAME
donwload_dataset(SELECTED_DATASET)
RAW_DATASET_DIRECTORY = DATASETS_ROOT_DIRECTORY_NAME / SELECTED_DATASET

CHOSEN_PREPROCESSES_LIST = [
    # PREPROCESS_CONVERT_TO_GRAY_SCALE,
    # PREPROCESS_CROP_TEXT_REGION,
    # PREPROCESS_RESIZE,
    # PREPROCESS_DESKEW,
    # PREPROCESS_NORMALIZE_PIXEL_VALUES,
    # PREPROCESS_DENOISING,
]

TEST_RATIO = 0.1
VALIDATION_RATIO = 0.2

EPOCHS_NUM = 10
BATCH_SIZE = 256
OPTIMIZER_NAME = OPTIMIZER_ADAM_NAME
OPTIMIZER_HYPERPARAMETERS_DICT = {
    # "momentum": 0.9,
    "weight_decay": 1e-3,
}
LEARNING_RATE = 0.001
SCHEDULER_NAME = SCHEDULER_PLATEAU_NAME
SCHEDULER_HYPERPARAMETERS_DICT = {
    "threshold": 0.01,
    "factor": 0.8,
    "patience": 4,
}
EARLY_STOP_METHODS = [EARLY_STOP_ACCURACY_LIMIT, EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT]
EARLY_STOP_METHODS_HYPERPARAMETERS = {
    ACCURACY_LIMIT: 0.998,
    LOSS_NO_IMPROVEMENT_EPOCH_LIMIT: 10,
}

TRAIN_BATCH_LOG_THRESHOLD = 10
TRAIN_EPOCH_AND_BATCH_EPOCH_SAMPLE_THRESHOLD = 1
TRAIN_EPOCH_AND_BATCH_BATCH_SAMPLE_THRESHOLD = 10

VALIDATION_BATCH_LOG_THRESHOLD = 10
VALIDATION_EPOCH_AND_BATCH_EPOCH_SAMPLE_THRESHOLD = 1
VALIDATION_EPOCH_AND_BATCH_BATCH_SAMPLE_THRESHOLD = 10

TEST_BATCH_LOG_THRESHOLD = 10
TEST_BATCH_SAMPLE_THRESHOLD = 10
#####################################################################
output_directories = []
data_functions_interface = DataFunctionsInterface(
    dataset_directory=RAW_DATASET_DIRECTORY,
    seed=MY_LUCKY_NUMBER,
    log_init_info=True,
)
output_directories.append(data_functions_interface._outputs_directory)
data_functions_interface.initial_preparation()
SPLITS_INFO_FILE_PATH = data_functions_interface.split_test_and_train_and_validation(
    test_ratio=TEST_RATIO,
    validation_ratio=VALIDATION_RATIO,
)
model = CNNAttentionOCR(
    vocab_size=len(VOCAB_LETTERS_INDEXES_DICT),
    input_channels=1,
    padding_index=VOCAB_LETTERS_INDEXES_DICT[PADDING_CHARACTER],
)
neural_network_functions_interface = NeuralNetworkFunctionsInterface(RAW_DATASET_DIRECTORY, data_functions_interface._outputs_directory, MY_LUCKY_NUMBER)
train_data_files_paths, validation_data_files_paths, test_data_files_paths = get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split(
    SPLITS_INFO_FILE_PATH
)
CHECKPOINT_DIRECTORY = neural_network_functions_interface.train(
    chosen_dataset_specific_preprocesses_name_list=[],
    chosen_dataset_specific_preprocesses_hyperparameters_dict={},
    chosen_common_preprocesses_list=[],
    chosen_common_preprocesses_hyperparameters_dict={},
    model=model,
    epochs_num=EPOCHS_NUM,
    train_data_files_names_list=train_data_files_paths,
    validation_data_files_names_list=validation_data_files_paths,
    batch_size=BATCH_SIZE,
    optimizer_name=OPTIMIZER_NAME,
    optimizer_hyperparameters_dict=OPTIMIZER_HYPERPARAMETERS_DICT,
    scheduler_name=SCHEDULER_NAME,
    scheduler_hyperparameters_dict=SCHEDULER_HYPERPARAMETERS_DICT,
    loss_function_name=CROSS_ENTROPY_LOSS,
    scores_name_list=ALL_SCORES_NAMES_LIST,
    early_stop_methods=EARLY_STOP_METHODS,
    early_stop_methods_hyperparameters=EARLY_STOP_METHODS_HYPERPARAMETERS,
    learning_rate=LEARNING_RATE,
    augmentation=None,
    train_batch_log_threshold=TRAIN_BATCH_LOG_THRESHOLD,
    train_epoch_and_batch_epoch_sample_threshold=TRAIN_EPOCH_AND_BATCH_EPOCH_SAMPLE_THRESHOLD,
    train_epoch_and_batch_batch_sample_threshold=TRAIN_EPOCH_AND_BATCH_BATCH_SAMPLE_THRESHOLD,
    validation_batch_log_threshold=VALIDATION_BATCH_LOG_THRESHOLD,
    validation_epoch_and_batch_epoch_sample_threshold=VALIDATION_EPOCH_AND_BATCH_EPOCH_SAMPLE_THRESHOLD,
    validation_epoch_and_batch_batch_sample_threshold=VALIDATION_EPOCH_AND_BATCH_BATCH_SAMPLE_THRESHOLD,
    log_init_info=True,
)

neural_network_functions_interface.test(
    model=model,
    test_data_files_names_list=test_data_files_paths,
    batch_size=BATCH_SIZE,
    scores_name_list=ALL_SCORES_NAMES_LIST,
    checkpoint_directory=CHECKPOINT_DIRECTORY,
    test_batch_log_threshold=TEST_BATCH_LOG_THRESHOLD,
    test_batch_sample_threshold=TEST_BATCH_SAMPLE_THRESHOLD,
    log_init_info=True,
)
