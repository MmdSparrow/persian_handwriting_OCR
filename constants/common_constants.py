############################################ SPLITS ############################################
SPLIT_TEST_NAME = "test"
SPLIT_TRAIN_NAME = "train"
SPLIT_VALIDATION_NAME = "validation"
SPLIT_TRAIN_AND_VALIDATION_NAME = "train_and_validation"
SPLITS_NAMES = [SPLIT_TEST_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME]

############################################ OPTIMIZERS ############################################
OPTIMIZER_SGD_NAME = "SGD"
OPTIMIZER_ADAM_NAME = "Adam"
OPTIMIZER_ADAMW_NAME = "AdamW"

############################################ OPTIMIZERS HYPERPARAMETERS ############################################
MOMENTUM_NAME = "momentum"
WEIGHT_DECAY_NAME = "weight_decay"

############################################ SCHEDULERS ############################################
SCHEDULER_CONSTANT_NAME = "ConstantLR"
SCHEDULER_PLATEAU_NAME = "ReduceLROnPlateau"

############################################ CHECKPOINT KEYS ############################################
BATCH_SIZE_KEY = "batch_size"
EPOCH_KEY = "epoch"
EPOCHS_NUM_KEY = "epochs_num"
MODEL_STATE_DICT_KEY = "model"
OPTIMIZER_NAME_KEY = "optimizer_name"
OPTIMIZER_STATE_DICT_KEY = "optimizer"
SCHEDULER_NAME_KEY = "scheduler_name"
SCHEDULER_STATE_DICT_KEY = "scheduler_state"
LOSS_NAME_KEY = "loss_name"
MIN_LOSS_KEY = "min_loss"
###
SCORES_KEY = "score"
LOSS_KEY = "loss"

############################################ SCORES ############################################
ACCURACY_SCORE_NAME = "Accuracy"
BALANCED_ACCURACY_SCORE_NAME = "Balalnced "
SENSITIVITY_SCORE_NAME = "Sensitivity"
SPECIFICITY_SCORE_NAME = "Specificity"
F1_SCORE_NAME = "F1-Score"
PRECISION_SCORE_NAME = "Precision"
AUC_SCORE_NAME = "AUC"

BALANCED_ACCURACY_NAME = "Balanced Accuracy"
MACRO_F1_SCORE_NAME = "Macro F1-Score"
WEIGHTED_F1_SCORE_NAME = "Weighted F1-Score"
MACRO_RECALL_NAME = "Macro Recall"
MACRO_PRECISION_NAME = "Macro Precision"

ALL_SCORES_NAMES_LIST = [
    ACCURACY_SCORE_NAME,
    SENSITIVITY_SCORE_NAME,
    SPECIFICITY_SCORE_NAME,
    F1_SCORE_NAME,
    PRECISION_SCORE_NAME,
    AUC_SCORE_NAME,
    BALANCED_ACCURACY_NAME,
    MACRO_F1_SCORE_NAME,
    WEIGHTED_F1_SCORE_NAME,
    MACRO_RECALL_NAME,
    MACRO_PRECISION_NAME,
]
PREDICTION_BASED_SCORES_NAMES_LIST = [
    ACCURACY_SCORE_NAME,
    SENSITIVITY_SCORE_NAME,
    SPECIFICITY_SCORE_NAME,
    F1_SCORE_NAME,
    PRECISION_SCORE_NAME,
]
BALANCED_DATASET_SCORES_NAMES_LIST = [
    ACCURACY_SCORE_NAME,
    SENSITIVITY_SCORE_NAME,
    SPECIFICITY_SCORE_NAME,
    F1_SCORE_NAME,
    PRECISION_SCORE_NAME,
    AUC_SCORE_NAME,
]
PROBABILITY_BASED_SCORES_NAMES_LIST = [
    AUC_SCORE_NAME,
]

############################################ LOSSES ############################################
CROSS_ENTROPY_LOSS = "cross_entropy"
CTC_LOSS = "ctc_loss"
FOCAL_LOSS = "focal_loss"

############################################ DATASETS ############################################
TYPED_WORDS_IMAGES_DATASET_NAME = "typed_words_images"

############################################ EARLY STOP ############################################
EARLY_STOP_ACCURACY_LIMIT = "early_stop_accuracy_limit"
ACCURACY_LIMIT = "accuracy_limit"
EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT = "early_stop_loss_no_improvement_epoch_limit"
LOSS_NO_IMPROVEMENT_EPOCH_LIMIT = "loss_no_improvement_epoch_limit"

############################################ PREPROCESS ############################################
PREPROCESS_CONVERT_TO_GRAY_SCALE = "convert to gray scale"
PREPROCESS_CROP_TEXT_REGION = "crop text region"
PREPROCESS_RESIZE = "resize"
PREPROCESS_DESKEW = "deskew"
PREPROCESS_NORMALIZE_PIXEL_VALUES = "normalize pixel values"
PREPROCESS_DENOISING = "denoising"

############################################ PREPROCESS HYPERPARAMETERS ############################################
PREPROCESS_HYPERPARAMETER_DENOISING_METHOD = "denoising_method"
PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_HEIGHT = "resizing_target_height"
PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_WIDTH = "resizing_target_width"

############################################ DENOISING METHODS ############################################
DENOISING_METHODS_MEDIAN_BLUR = "median blur"

############################################ PREPROCESSES ############################################

CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY = []
CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT = []