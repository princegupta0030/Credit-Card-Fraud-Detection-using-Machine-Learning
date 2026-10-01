# Credit Card Fraud Detection

This is an end-to-end Machine Learning project for detecting credit card fraud. It aims to be beginner-friendly, properly structured, easy to explain in an interview, and reproducible.

## Project Structure

```
credit-card-fraud-detection/
│
├── data/                  <- Data directory
│   ├── raw/               <- The original, immutable data dump.
│   └── processed/         <- The final, canonical data sets for modeling.
│
├── notebooks/             <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description, e.g.
│                             `1.0-jqp-initial-data-exploration.ipynb`.
│
├── src/                   <- Source code for use in this project.
│   ├── __init__.py        <- Makes src a Python module.
│   ├── data_preprocessing.py <- Scripts to download or generate data and clean it.
│   ├── feature_engineering.py<- Scripts to turn raw data into features for modeling.
│   ├── train.py           <- Scripts to train models.
│   ├── evaluate.py        <- Scripts to evaluate model performance (metrics, etc.).
│   └── predict.py         <- Scripts to run predictions on new data.
│
├── models/                <- Trained and serialized models, model predictions, or model summaries.
│
├── reports/               <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/           <- Generated graphics and figures to be used in reporting.
│
├── app.py                 <- Streamlit application for interacting with the model and data.
├── requirements.txt       <- The requirements file for reproducing the analysis environment.
├── README.md              <- The top-level README for developers using this project.
└── .gitignore             <- Specifies intentionally untracked files to ignore.
```
