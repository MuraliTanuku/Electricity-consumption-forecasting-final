# Electricity Consumption Forecasting Using Time-Series Analysis

A complete Python machine-learning project with a browser dashboard (HTML, CSS and JavaScript). The dashboard calls a local Flask API so the charts and scores come from the actual Python models, not a JavaScript-only approximation.

## Files

- `app.py` — Flask web server and API
- `model_engine.py` — data preparation, feature engineering, model comparison, evaluation and recursive forecasting
- `electricity_forecasting.py` — command-line entry point
- `electricity_consumption.csv` — supplied sample dataset
- `static/` — HTML, CSS and JavaScript frontend
- `outputs/` — generated CSV results after a run
- `original_electricity_forecasting.py` — original starter script preserved for reference

## Run the project

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the dashboard:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in your browser.
6. Click **Run model comparison**. You can also upload a CSV with `Date` and `Consumption_kWh` columns.

To run without the browser:

```bash
python electricity_forecasting.py
```

## Models compared

- Naive persistence (next value equals the latest observed value)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Tuned Random Forest using `TimeSeriesSplit` cross-validation on the training segment

## Features

Lag features at 1, 2, 7, 14 and 28 days; 7-day and 30-day rolling means; day of week, month and day of year. A chronological 80/20 split is used. The same test period is used for all model comparisons.

## Performance measures

MAE, MSE, RMSE, R² and MAPE. Lower MAE/MSE/RMSE/MAPE is better; higher R² is better. MAPE excludes zero-valued actual observations to avoid division by zero. The model with the lowest test RMSE is selected for the 30-day recursive forecast.

## Outputs

After a successful run, `outputs/model_comparison.csv` and `outputs/future_forecast.csv` are created. Download links are also available in the dashboard.

## Important notes for evaluation

- The included CSV is the sample dataset supplied with the starter project. Replace it with real meter readings for a real-world evaluation.
- Scores are calculated when you run the experiment; they are intentionally not hard-coded or fabricated.
- The tuned Random Forest uses only the training portion for cross-validation. The final test segment is held out for comparison.
- This is an applied project contribution combining time-series feature engineering, model benchmarking, hyperparameter tuning and recursive forecasting. Do not describe it as a newly invented ML algorithm.
- The dashboard loads Chart.js from a CDN, so an internet connection is required for charts. The Python backend and model training run locally.
