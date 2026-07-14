# Wine Quality Prediction — Ordinal Multiclass Classification

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org)

Supervised and unsupervised machine learning analysis of a wine quality dataset. The project examines whether physicochemical properties can classify wines into three ordinal quality groups: **low quality**, **medium quality** and **high quality**.

> **Responsible-use note:** This is an academic machine learning project. It is not a production-ready wine quality assessment tool and should not be used for commercial, laboratory or regulatory decisions without further validation.

---

## Business question

Can physicochemical properties of wine help classify its perceived quality?

The dataset contains **1,599 wines** described by **11 physicochemical variables**, including:

- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol

The original target variable `quality` ranges from 3 to 8. Since it is discrete and ordinal, it is transformed into three classes:

- **Low quality (`0`)**: original quality 3 or 4
- **Medium quality (`1`)**: original quality 5 or 6
- **High quality (`2`)**: original quality 7 or 8

The resulting target is highly imbalanced:

- **Low quality (`0`)**: 63 wines (**3.94%**)
- **Medium quality (`1`)**: 1,319 wines (**82.49%**)
- **High quality (`2`)**: 217 wines (**13.57%**)

This imbalance is a central modelling issue: a classifier can obtain high accuracy by predicting mostly `medium quality`.

---

## What the notebook does

1. **Data inspection** — loads the dataset and studies the original variables  
2. **Exploratory analysis** — analyses correlations between physicochemical variables and wine quality  
3. **Target transformation** — converts the original `quality` score into three ordinal classes  
4. **Preprocessing** — uses a stratified 80/20 train/test split and avoids scaling leakage  
5. **Unsupervised analysis** — applies PCA and K-Means to explore latent wine profiles  
6. **Supervised learning** — compares:
   - K-Nearest Neighbours
   - Logistic Regression
   - Support Vector Machine
   - Random Forest
7. **Hyperparameter tuning** — uses stratified cross-validation on the training data  
8. **Model comparison** — evaluates accuracy, balanced accuracy, macro F1-score, weighted F1-score, classification reports and confusion matrices  
9. **Interpretation** — discusses why accuracy alone is not enough for this imbalanced multiclass problem  

---

## Exploratory findings

The correlation analysis suggests that `quality` is not explained by a single variable. The strongest positive relationship with quality appears for `alcohol`, while `volatile acidity` has a negative relationship with quality.

Other variables such as `sulphates` and `citric acid` show weaker positive associations. The dataset also contains correlations between predictors, for example between acidity-related variables and between free and total sulfur dioxide.

---

## Unsupervised findings

PCA and K-Means are used only for exploratory analysis. The target variable `processed_quality` is not used to build the components or clusters.

The best silhouette score among the tested K-Means solutions is obtained at **k = 2**. The two clusters contain:

- **Cluster 0**: 590 wines
- **Cluster 1**: 1,009 wines

The variables that most differentiate the clusters are:

- `total sulfur dioxide`
- `free sulfur dioxide`
- `fixed acidity`
- `residual sugar`
- `citric acid`
- `alcohol`

The clusters reveal two broad physicochemical wine profiles, but they should be interpreted as descriptive groups rather than direct quality predictions.

---

## Supervised results

All models are evaluated on the same held-out test set.

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| SVM | 0.7750 | **0.6865** | **0.5892** | 0.7975 |
| Random Forest | **0.8563** | 0.5975 | 0.5749 | **0.8490** |
| KNN | 0.8469 | 0.5353 | 0.5439 | 0.8364 |
| Logistic Regression | 0.8438 | 0.4577 | 0.4676 | 0.8168 |

### Key interpretation

Random Forest achieves the highest overall accuracy, but this metric is strongly influenced by the majority class `medium quality`.

SVM provides the best **balanced accuracy** and **macro F1-score**, which makes it more suitable when the goal is to treat all classes more fairly. This is important because the dataset is highly imbalanced and the minority class `low quality` is difficult to detect.

Logistic Regression works as an interpretable baseline, but it struggles with the minority class. KNN performs reasonably well overall, although it is also affected by the dominance of the medium-quality class.

The main lesson is that **accuracy alone is not enough** for this problem. Macro F1-score and balanced accuracy provide a more honest evaluation of model performance across all quality groups.

---

## Stack

- **Python** — NumPy, Pandas  
- **Scikit-learn** — preprocessing, PCA, K-Means, model training, tuning and metrics  
- **Matplotlib + Seaborn** — visualisation  
- **Jupyter Notebook** — reproducible analysis  

---

## How to run

Place `winequality.csv` in the same directory as the notebook.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook wine_quality_prediction.ipynb
```

---

## Project structure

```text
├── wine_quality_prediction.ipynb   # Main analysis notebook
├── winequality.csv                 # Input dataset
└── README.md
```
