# AI-Driven PM2.5 Forecasting and AQI Classification in Singapore

## Overview
This project uses AI to forecast PM2.5 levels and classify AQI in Singapore, contributing to UNSDGs 11 (Sustainable Cities) and 13 (Climate Action). It processes hourly data, performs EDA, models with RF/LSTM, evaluates, and compresses models for efficiency.

## Dependencies
- Python 3.8+
- Libraries: pandas, numpy, requests, matplotlib, seaborn, scikit-learn, tensorflow, statsmodels, shap, joblib
- Install via: `pip install -r requirements.txt` (create requirements.txt with `pip freeze > requirements.txt` after setup)

## How to Run
Run notebooks sequentially in Jupyter (e.g., `jupyter notebook`):
1. **Data Loading and Preprocessing.ipynb**: Fetches/merges data, saves `sensor_..._capped.csv`.
2. **Data Exploratory Analysis.ipynb**: Loads CSV, generates plots/CSVs (e.g., correlations.csv).
3. **AI Modelling.ipynb**: Loads processed data, adds features, trains/saves models (e.g., rf_model_h1.pkl, lstm_model_h1.h5) and `featured_data.csv`.
4. **Assessment and Evaluation.ipynb**: Loads models/data, evaluates, saves `evaluation_results.csv`.
5. **Model Compression.ipynb**: Loads models/test data, compresses for h=6, saves TFLite/Joblib files and `compression_results.csv`.

## Reproducibility
- Set random_state=42 where applicable.
- Use provided API key for OpenAQ (replace in Notebook 1).
- Data range: 2022-05-01 to 2024-04-30.
- Outputs are saved as CSVs/models for inspection.
- Run on CPU/GPU; TensorFlow may vary slightly by hardware, but results are consistent within ~1%.