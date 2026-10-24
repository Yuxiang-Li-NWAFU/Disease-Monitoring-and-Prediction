# Wheat Stripe Rust Prediction Code

This repository contains the five Python scripts used for data preparation, preprocessing, feature selection, model training, and Figure 4 generation for the wheat stripe rust prediction study.

The source dataset and all intermediate datasets are intentionally not included. The code requires a locally available input workbook before it can be run end to end.

## Input data

Place the private input workbook at:

`data/raw_data.xlsx`

The workbook must contain a worksheet named `Sheet1` and a target column named `disease_level`. The preprocessing script converts the target to a binary outcome using the existing rule: levels 0–2 are mapped to class 0 and levels above 2 are mapped to class 1.

## Execution order

Run the scripts from the repository directory in the following order:

1. `python 01_data_loader.py`
2. `python 02_preprocessing.py`
3. `python 03_feature_selection.py`
4. `python 04_model_training.py`
5. `python 05_plot_figure4.py`

The scripts use a shared `results/` directory to pass intermediate files between stages. The model-training script writes its results to a timestamped directory under `results/model_comparison_*/`. The Figure 4 script locates the most recently modified `Table_S12_generated.xlsx` in those results directories and saves the PNG and PDF figures to `results/figures/`.

## Dependencies

Install the packages listed in `requirements.txt` in a Python environment compatible with the installed versions of scikit-learn, imbalanced-learn, SHAP, LightGBM, and XGBoost.

## Data and generated files

The repository does not distribute the raw workbook, processed CSV files, selected-feature files, fitted preprocessing objects, model outputs, or generated figures. These files are created locally when the scripts are run. The `data/raw_data.xlsx` file and the contents of `results/` are excluded from version control by `.gitignore`.

The scripts have been syntax-checked. Full execution requires the original input workbook, which is not included in this repository.
