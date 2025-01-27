This repository contains a Python script designed to process and prepare crime data from São Paulo's Department of Public Safety. The script converts crime data into a graph-based format suitable for analysis using graph neural networks, with specific focus on generating temporal features, adjacency matrices, and node labels. The processed data can be used for training models like Dynamic Self-Attention Networks (DySAT) and baseline methods. The core components of this repository include:

1. **Data Loader Script:** This script processes crime data and generates the necessary datasets for training and evaluation. It prepares the data in a format that is compatible with the Dynamic Self-Attention Network (DySAT) model and baseline machine learning methods.  
   *Note: The input to the data loader is derived from the output of our previously detailed data pipeline. For more information, please refer to the [data pipeline subdirectory](https://github.com/giva-lab/BRACIS2024_GNN/tree/4935f2755d77af2fcb356bf504ce99f760d4ea24/crime-data).*

2. **DySAT Model Script:** This script is used to run the DySAT model, which leverages dynamic graph structures to generate robust node embeddings.

3. **Baseline Methods Script:** This script runs traditional machine learning methods, including Random Forest, SVM, and Logistic Regression, to compare their performance with the DySAT model.

---

# Crime Data Processing and Feature Extraction

This repository contains a Python script designed to process and prepare crime data from São Paulo's Department of Public Safety. The script converts crime data into a graph-based format suitable for analysis using graph neural networks, with specific focus on generating temporal features, adjacency matrices, and node labels. The processed data can be used for training models like Dynamic Self-Attention Networks (DySAT) and Evolving Graph Convolutional Networks (EvolveGCN).

## Usage

### Run Data Loader Script

```bash
python dataloader_Dysat.py --process_id your_data_id --time_delta M --time_steps 31 --window -1 --overwrite
```
### Command-Line Arguments

- `--process_id`, `-id` (str): Identifier for the processed data. This should match the folder name where processed data is stored using the data pipeline.
- `--time_delta`, `-td` (str): Time period to consider when grouping crimes. Options are 'D' (day), 'M' (month), or 'Y' (year). The default is 'M'.
- `--overwrite` (flag): If set, existing data will be overwritten.
- `--time_steps` (int): Total time steps used for training, evaluation, and testing. Default is 31.
- `--window` (int): Window size for temporal attention. Default is -1, which means full window.

## Output

The script generates several pickle files in the specified output directory:

- `nodes_labels_times__<time_delta>.pickle`: Node labels and time-step information.
- `graphs_all_times__<time_delta>.pickle`: List of graph objects for each time step.
- `features_all_times__<time_delta>.pickle`: Feature matrices for each time step.
- `features_analysis__<time_delta>.pickle`: Feature matrices used for feature analysis.
- `adjacency_all_times__<time_delta>.pickle`: Adjacency matrices for each time step.
- `dataset_dynamic__<time_delta>.pickle`: Dataset object ready for training with graph neural networks.


---

# DySAT Model Training Script

The Dynamic Self-Attention Network (DySAT) is a powerful model that leverages dynamic graph structures to generate robust node embeddings for various downstream tasks, such as crime prediction. This script handles the complete workflow, from data loading and preprocessing to model training and evaluation.

## Requirements

To run this code, you will need the following Python packages:

- torch
- numpy
- pandas
- argparse
- pickle
- pathlib
- logging

## Installation

1. Clone this repository.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To run the DySAT model, use the following command:

```bash
python train_dysat.py --time_delta [D|M|Y] [other options]
```

### Arguments

- `--time_delta` (str, required): Time period for grouping crimes by Year (`Y`), Day (`D`), or Month (`M`). Default is Month (`M`).
- `--time_steps` (int): Total time steps used for training, evaluation, and testing. Defaults based on the `time_delta`.
- `--epochs` (int): Number of training epochs. Default is 5000.
- `--batch_size` (int): Batch size for training. Default is 32.
- `--early_stop` (int): Number of epochs to wait before stopping early. Default is 30.
- `--learning_rate` (float): Initial learning rate. Default is 0.01.
- `--spatial_drop` (float): Dropout rate for spatial attention. Default is 0.4.
- `--temporal_drop` (float): Dropout rate for temporal attention. Default is 0.4.
- `--weight_decay` (float): Weight decay for the optimizer. Default is 0.0005.
- `--class_weight` (float): Class weight for the minority class. Default is 4.420.

Refer to the script for additional tunable hyperparameters and architecture configurations.

## Training

The training process involves the following steps:

1. **Data Loading:** The script loads pre-processed data from the specified path. Data is divided into training, validation, and test sets.
2. **Model Initialization:** The DySAT model is initialized with the specified parameters.
3. **Training Loop:** The model is trained over a number of epochs, with training and validation losses recorded.
4. **Validation and Testing:** After each epoch, the model is validated on a separate set of data, and testing is done on the test set to evaluate performance.
5. **Checkpointing:** The best model based on validation performance is saved.

## Model Checkpointing

The model is periodically saved during training based on its performance on the validation set. The best model is saved at:

```plaintext
./Results/static_features/model_checkpoints/model_best.pth.tar
```

## Logging and Metrics

Training and evaluation metrics are logged for each epoch. These logs include:

- Training loss
- Validation and test precision, recall, F1-score, AUC, MCC, and balanced accuracy
- Misclassification rates
- True/False positive/negative counts

Logs are saved in JSON format for easy analysis:

```plaintext
./Results/static_features/model_checkpoints/training_logs_manual.json
```

## Results
Upon training completion, the model will display performance metrics on the test set, including:

- Precision
- Recall
- F1-score
- AUC
- Balanced accuracy
- Misclassification rate

These metrics help in assessing the effectiveness of the model in predicting crime patterns.

---

# Baseline Methods for Binary Classification

This repository contains code to run traditional machine learning models as baseline methods for binary classification tasks. The code preprocesses temporal features, splits the data into training and testing sets, trains the model, and evaluates it using various performance metrics.

## Features
- **Preprocessing:** Temporal features are extracted and normalized.
- **Models Supported:** Logistic Regression, Random Forest, SVM, and Decision Trees.
- **Evaluation Metrics:** Precision, Recall, F1 Score, AUC, Accuracy, Matthews Correlation Coefficient (MCC), Balanced Accuracy, and Misclassification Rate.

## Requirements

- Python 3.12
- Pandas
- NumPy
- Scipy
- scikit-learn
- Matplotlib
- pathlib
- pickle

## Usage

### Command Line Arguments

- `--time_delta`: Time period for grouping crimes (Year, Day, or Month). Choices are "D", "M", "Y". Default is "M".
- `--model`: The machine learning model to use for classification. Choices are "logistic", "randomforest", "svm", "decisiontrees". Default is "logistic".

### Example Usage

```bash
python run_baseline_methods.py --time_delta M --model randomforest
```

### Steps in the Code

1. **Load Data**:
    - Data is loaded from a specified directory containing the preprocessed features.
  
2. **Preprocess Features**:
    - The feature matrix is row-normalized, and labels are extracted.
  
3. **Generate Temporal Features**:
    - For each time period, temporal features are extracted and combined with static features.
  
4. **Train-Test Split**:
    - Data is split into training and testing sets using a sliding window approach.
  
5. **Model Training**:
    - The specified machine learning model is trained on the training data.
  
6. **Model Evaluation**:
    - The model is evaluated on the testing data, and various metrics are calculated and printed.

### Output

The code will output the following:
- **Model Probabilities**: Saved as a CSV file named `<model>_output_probs_Dyanmic_features.csv`.
- **Performance Metrics**: Printed to the console, including precision, recall, F1 score, AUC, accuracy, MCC, balanced accuracy, and misclassification rate for both positive and negative classes.

## Directory Structure

```
├── data/
│   └── dataloader/
│       └── dysat/
│           └── label_column_static_feats/
│               └── <time_period>/
│                   └── features_all_times__<time_delta>.pickle
├── dataloader/
│   └── minibatch.py
├── utils.py
└── run_baseline_methods.py
```

## Notes

- The script assumes that the data is preprocessed and stored as pickled sparse matrices.
- The data loader module (`minibatch.py`) should be properly configured in the `dataloader` directory.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

   
