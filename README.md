# Smartphone Human Activity Recognition

## Project idea

This project recognizes a person's physical activity from a short window of smartphone motion sensor data. It compares three pattern-recognition approaches on the same six-class task:

1. **Random Forest** trained on a selected subset of the dataset's engineered features.
2. **Dense neural network (DNN)** trained on the same selected features.
3. **1D convolutional neural network (time-series CNN)** trained directly on raw inertial signal windows.

The experiment studies both model choice and input representation: compact, hand-engineered feature vectors versus raw accelerometer and gyroscope sequences.

## Dataset

The repository contains the UCI Human Activity Recognition Using Smartphones (UCI HAR) dataset. Thirty volunteers aged 19–48 performed six activities while wearing a Samsung Galaxy S II smartphone at the waist. The phone's accelerometer and gyroscope were sampled at 50 Hz.

| Property | Description |
|---|---|
| Activities | Walking, walking upstairs, walking downstairs, sitting, standing, laying |
| Window duration | 2.56 seconds |
| Samples per raw window | 128 time steps |
| Overlap between windows | 50% |
| Engineered feature vector | 561 time and frequency domain features |
| Training set | 7,352 windows from 21 subjects |
| Test set | 2,947 windows from 9 different subjects |
| Raw signal channels | 9: body acceleration, body gyroscope, total acceleration on x/y/z axes |

The official subject-separated train/test split is retained. The feature selector is fitted only on training samples, then applied to both sets, preventing test-label information from affecting feature selection. The dataset's precomputed features are normalized to the range [-1, 1].

Data files are under `dataset/`. See `dataset/README.txt` and `dataset/features_info.txt` for the source's full description and feature definitions. The dataset README requests citation of the original publication and states that commercial use is prohibited.

## Methods

### 1. Feature selection

`01_feature_selection.py` uses scikit-learn's ANOVA F-test (`SelectKBest` with `f_classif`) to retain the top **100** features by default. Set a different count with `--k`; for example, `--k 75`. It saves the selected train/test arrays, labels, feature names, and original feature indices to `outputs/selected_features.npz`.

### 2. Random Forest

`02_random_forest.py` fits a class-weighted Random Forest with 400 trees using the selected features. The tree count can be changed with `--trees`.

### 3. Dense neural network

`03_dnn.py` fits a dense network on the same selected feature vectors. Its hidden layers have 256, 128, and 64 units with ReLU activations, batch normalization, and dropout. Early stopping monitors validation loss.

### 4. Raw time-series CNN

`04_time_series_cnn.py` loads the nine raw inertial signal files and stacks each sample into a 128-by-9 array. Its 1D CNN learns temporal patterns from those sensor windows directly. It does not use the 100-feature subset; it uses the same train/test subjects and activity labels as the other models.

All three models are evaluated against the official test set. The confusion matrix convention is actual activity in rows and predicted activity in columns.

## Results

The following test results are recorded in the current `outputs/` directory. Each model was evaluated on the same 2,947 held-out windows.

| Approach | Input | Test accuracy | Macro F1 |
|---|---|---:|---:|
| Random Forest (400 trees) | 100 selected features | 88.84% | 0.8863 |
| Dense neural network | 100 selected features | **90.40%** | **0.9030** |
| Time-series 1D CNN | Raw 128 × 9 windows | 83.41% | 0.8307 |

In these saved runs, the dense network achieved the highest accuracy and macro F1. The time-series CNN did not outperform the models using engineered features. This is an observed result for the current architecture and training setup, not a general conclusion that engineered features are always better. Per-class precision, recall, F1, and confusion matrices are saved for further analysis.

## Requirements

- Python 3.12.7
- Dependencies listed in `requirements.txt` (NumPy, scikit-learn, TensorFlow)

## Setup and run

From the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the environment's interpreter directly using `.\.venv\Scripts\python.exe` in place of `python`.

Run the pipeline in order:

```powershell
python 01_feature_selection.py --k 100
python 02_random_forest.py
python 03_dnn.py
python 04_time_series_cnn.py
```

The first script must run before Random Forest and DNN because they use the selected-feature archive. The time-series CNN reads the raw signal files directly. Training scripts can take optional `--epochs` and `--batch-size` arguments for the neural networks.

To display the selected feature count/names and each model's test accuracy and labeled confusion matrix, run:

```powershell
python view_results.py
```

Or double-click `view_results.bat` on Windows. Use `python view_npz.py` if you want to inspect the arrays in `outputs/selected_features.npz`.

To visualize all three models' test accuracy, macro F1, and labeled confusion matrices in one figure, run:

```powershell
python visualize_results.py
```

The figure is saved to `outputs/model_comparison.png` and shown in a window. Add `--no-show` to save without opening a window.

## Generated outputs

| Path | Contents |
|---|---|
| `outputs/selected_features.npz` | Selected `X_train`, `X_test`, `y_train`, `y_test`, selected feature names and indices |
| `outputs/random_forest/` | Random Forest metrics and confusion matrix |
| `outputs/dnn/` | DNN metrics, confusion matrix, and saved Keras model |
| `outputs/time_series_cnn/` | CNN metrics, confusion matrix, and saved Keras model |
| `outputs/model_comparison.png` | Combined performance and confusion-matrix visualization |

Metric files are JSON; confusion matrices are CSV. Re-running scripts overwrites the corresponding outputs.

## Conclusion

The project provides a controlled comparison of a classical ensemble, a dense neural model, and a temporal CNN for smartphone activity recognition. In the current run, the DNN on the selected 100 engineered features performed best on the subject-separated test set (90.40% accuracy). The subject split makes the evaluation more meaningful for unseen people. Future work could tune the number of selected features, compare feature-selection methods, improve temporal modeling, and analyze activities that the confusion matrices show are often mistaken for one another.

## Dataset citation

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. “A Public Domain Dataset for Human Activity Recognition Using Smartphones.” *21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN)*, 2013.
