<h1 align="center">Persian Handwritten OCR Framework</h1>

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.x-blue.svg">
    <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg">
    <img src="https://img.shields.io/badge/OCR-Handwritten%20Recognition-green.svg">
    <img src="https://img.shields.io/badge/Language-Persian-red.svg">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
</p>

A modular **Persian Handwritten Text Recognition (HTR) framework** built with **Python** and **PyTorch**. The project provides a flexible infrastructure for developing, training, validating, and evaluating different deep learning architectures for Persian handwritten text recognition.

Unlike traditional OCR implementations where training and evaluation pipelines must be implemented separately for each model, this framework provides a unified API that handles the entire machine learning workflow. Developers can integrate new neural network architectures without having to reimplement training, validation, metric calculation, or result analysis.

> **Main Idea:** Focus on designing better OCR architectures while the framework handles the rest of the machine learning pipeline.

## Overview

The framework provides an end-to-end workflow for Persian handwritten text recognition:

1. **Dataset Management** – Automatically download and prepare supported datasets.
2. **Data Preprocessing** – Apply configurable image preprocessing operations.
3. **Dataset Splitting** – Split datasets into training, validation, and test sets.
4. **Model Integration** – Integrate different neural network architectures through a unified interface.
5. **Model Training** – Train models using configurable optimizers, loss functions, learning rate schedulers, and other hyperparameters.
6. **Validation** – Automatically validate models and calculate evaluation metrics.
7. **Early Stopping** – Stop training based on configurable termination conditions.
8. **Model Evaluation** – Evaluate trained models using multiple metrics.
9. **Cross-Validation** – Perform K-Fold cross-validation and analyze results across folds.
10. **Result Analysis** – Analyze training outputs, compare experiments, and generate visualization plots.

## Key Features

### 🔹 Modular Model Integration

One of the main objectives of this framework is to make integrating new OCR architectures straightforward.

- Integrate different neural network architectures without implementing training or validation loops.
- Unified interface for different model architectures.
- Support for CNN-based and attention-based OCR architectures.
- Model-independent training and evaluation pipelines.
- Easily experiment with different architectures under identical training conditions.

### 🔹 Dataset Management

- Automatic dataset downloading.
- Support for different handwritten OCR datasets.
- Centralized dataset management.
- Reproducible dataset splitting using configurable random seeds.
- Separate training, validation, and test datasets.

### 🔹 Image Preprocessing

Supports configurable preprocessing operations, including:

- Grayscale conversion.
- Text region cropping.
- Image resizing.
- Image deskewing.
- Pixel value normalization.
- Image denoising.
- Dataset-specific preprocessing operations.
- Common preprocessing operations shared across datasets.

### 🔹 Configurable Training Pipeline

The framework provides a unified training API with configurable components.

**Supported optimizers:**
- Adam
- Other configurable PyTorch optimizers.

**Supported loss functions:**
- CTC Loss
- Cross Entropy Loss
- Other configurable loss functions.

**Supported learning rate schedulers:**
- Constant learning rate.
- ReduceLROnPlateau.
- Other configurable PyTorch learning rate schedulers.

**Early Stopping:**
- Accuracy threshold.
- Loss improvement monitoring.
- Configurable stopping conditions and hyperparameters.

Additional training capabilities:

- Configurable batch size and number of epochs.
- Optimizer hyperparameter configuration.
- Learning rate configuration.
- Weight decay support.
- Data augmentation support.
- Configurable training and validation logging.
- Model checkpoint management.

### 🔹 Evaluation Metrics

The framework supports multiple evaluation metrics that can be selected through the training API.

Instead of manually implementing metric calculations for every model, developers can simply specify the desired metrics.

For handwritten text recognition, supported evaluation metrics can include:

- Character Error Rate (CER).
- Word Error Rate (WER).
- Accuracy.
- Precision.
- Recall.
- F1-score.

Metrics can be extended to support additional evaluation requirements.

### 🔹 K-Fold Cross-Validation

The framework supports K-Fold cross-validation for more reliable model evaluation.

- Automated K-Fold execution.
- Independent training and validation for each fold.
- Configurable number of folds.
- Consistent evaluation across folds.
- Aggregation of evaluation metrics.
- Statistical analysis of cross-validation results.
- Comparison of model performance across folds.

### 🔹 Training and Result Analysis

The framework includes tools for analyzing model training and evaluation outputs.

**Training Analysis:**
- Training and validation loss analysis.
- Evaluation metric tracking across epochs.
- Training performance visualization.
- Experiment output analysis.

**Cross-Validation Analysis:**
- Analyze results from individual folds.
- Calculate mean and standard deviation of evaluation metrics.
- Compare model performance across folds.
- Visualize cross-validation results.

**Visualization:**
- Training and validation loss plots.
- Evaluation metric plots.
- Cross-validation performance diagrams.
- Statistical result visualizations.

## Usage

The framework is designed around a simple API that separates model implementation from the training infrastructure.

### 1. Dataset Configuration

Select a dataset. Supported datasets can be downloaded automatically.

```python
SELECTED_DATASET = TYPED_WORDS_IMAGES_DATASET_NAME

RAW_DATASET_DIRECTORY = (
    DATASETS_ROOT_DIRECTORY_NAME / SELECTED_DATASET
)
```

### 2. Configure Training Parameters

Configure preprocessing, dataset splitting, optimizer, scheduler, loss function, and evaluation metrics.

```python
TEST_RATIO = 0.1
VALIDATION_RATIO = 0.2

EPOCHS_NUM = 10
BATCH_SIZE = 256

OPTIMIZER_NAME = OPTIMIZER_ADAM_NAME

OPTIMIZER_HYPERPARAMETERS_DICT = {
    "weight_decay": 1e-3,
}

LEARNING_RATE = 0.001

SCHEDULER_NAME = SCHEDULER_PLATEAU_NAME

SCHEDULER_HYPERPARAMETERS_DICT = {
    "threshold": 0.01,
    "factor": 0.8,
    "patience": 4,
}

EARLY_STOP_METHODS = [
    EARLY_STOP_ACCURACY_LIMIT,
    EARLY_STOP_LOSS_NO_IMPROVEMENT_EPOCH_LIMIT,
]

EARLY_STOP_METHODS_HYPERPARAMETERS = {
    ACCURACY_LIMIT: 0.998,
    LOSS_NO_IMPROVEMENT_EPOCH_LIMIT: 10,
}
```

### 3. Dataset Preparation

Initialize the dataset interface and prepare the dataset.

```python
data_functions_interface = DataFunctionsInterface(
    dataset_directory=RAW_DATASET_DIRECTORY,
    seed=MY_LUCKY_NUMBER,
    log_init_info=True,
)

data_functions_interface.initial_preparation()

SPLITS_INFO_FILE_PATH = (
    data_functions_interface
    .split_test_and_train_and_validation(
        test_ratio=TEST_RATIO,
        validation_ratio=VALIDATION_RATIO,
    )
)
```

### 4. Define Your OCR Model

Implement your own neural network architecture and pass it to the framework.

For example:

```python
model = CNNAttentionOCR(
    vocab_size=len(VOCAB_LETTERS_INDEXES_DICT),
    input_channels=1,
    padding_index=VOCAB_LETTERS_INDEXES_DICT[
        PADDING_CHARACTER
    ],
)
```

The framework handles the training and validation processes independently of the specific architecture.

### 5. Train the Model

Initialize the neural network interface and start training.

```python
neural_network_functions_interface = (
    NeuralNetworkFunctionsInterface(
        RAW_DATASET_DIRECTORY,
        data_functions_interface._outputs_directory,
        MY_LUCKY_NUMBER,
    )
)

CHECKPOINT_DIRECTORY = (
    neural_network_functions_interface.train(
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
        early_stop_methods_hyperparameters=(
            EARLY_STOP_METHODS_HYPERPARAMETERS
        ),

        learning_rate=LEARNING_RATE,

        augmentation=None,

        log_init_info=True,
    )
)
```

The framework automatically manages the training and validation process, loss calculation, evaluation metrics, learning rate scheduling, early stopping, logging, and checkpoint generation.

*Note: The example above focuses on the main configuration parameters. Logging thresholds and sampling parameters can also be configured through the training API.*

## Architecture Philosophy

The framework follows a modular design that separates the following responsibilities:

| Component | Responsibility |
|---|---|
| Dataset Management | Downloading, preparing, and splitting datasets |
| Preprocessing | Image preprocessing and augmentation |
| Model Architecture | Neural network implementation |
| Training Engine | Model training and optimization |
| Validation Engine | Validation and metric calculation |
| Evaluation | Model performance evaluation |
| Cross-Validation | K-Fold execution and result aggregation |
| Analysis | Statistical analysis and visualization |

This separation makes the framework extensible and allows researchers to experiment with different architectures, training configurations, and evaluation strategies without repeatedly implementing the underlying infrastructure.

## Future Improvements

Potential directions for further development include:

- 🚀 Support for additional OCR architectures.
- 📚 Integration of more Persian handwritten datasets.
- 🔍 Advanced hyperparameter optimization.
- 📊 More evaluation metrics and statistical analysis methods.
- ⚡ Distributed and mixed-precision training.
- 🧪 Additional cross-validation strategies.
- 🔄 Automated experiment comparison.
- 📈 More comprehensive result visualization tools.

## Future Vision

The long-term goal is to develop a comprehensive and extensible framework for Persian handwritten text recognition research.

By providing a unified training and evaluation infrastructure, the framework aims to reduce repetitive implementation work and make it easier to develop, compare, and evaluate different deep learning architectures.

The ultimate objective is to allow researchers and developers to focus on what matters most: **building better models for Persian handwritten text recognition.**