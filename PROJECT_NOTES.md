# Project methodology and viva notes

## Problem
Electricity demand varies over time. Forecasting future daily consumption can help plan usage and manage energy resources.

## Workflow
1. Read the CSV and validate `Date` and `Consumption_kWh`.
2. Convert readings to a daily series and interpolate missing calendar days.
3. Create lag, rolling-average and calendar features.
4. Split observations chronologically: first 80% for training, final 20% for testing.
5. Compare a persistence baseline and five machine-learning approaches.
6. Tune Random Forest using expanding-window time-series cross-validation on training data only.
7. Compare MAE, MSE, RMSE, R² and MAPE; choose the lowest-RMSE model.
8. Generate a recursive 30-day forecast and export the results.

## Project contribution / novelty
The project contribution is the integration of lag and rolling-window features, calendar features, multiple-model benchmarking, time-aware hyperparameter tuning and recursive future prediction in one workflow. This is a practical system-level contribution, not a claim that a new algorithm has been invented.

## Possible further improvements
- Add temperature, humidity, holidays and tariff information if reliable matching data becomes available.
- Evaluate with rolling-origin backtesting across multiple test windows.
- Compare SARIMA/Prophet or an LSTM when there is enough historical data.
- Monitor forecast errors as new actual readings arrive.
