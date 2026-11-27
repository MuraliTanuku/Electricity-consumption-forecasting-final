"""
Electricity Consumption Forecasting Using Time-Series Analysis
---------------------------------------------------------------
Run:
    pip install -r requirements.txt
    python electricity_forecasting.py

Input CSV format:
    Date,Consumption_kWh
    2022-01-01,125.50

The included CSV is synthetic sample data for demonstration. Replace it with
your own meter data using the same two column names for a real-world study.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
DATA_FILE = "electricity_consumption.csv"
OUTPUT_DIR = "outputs"
FORECAST_DAYS = 30
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(path):
    data = pd.read_csv(path)
    required = {"Date", "Consumption_kWh"}
    if not required.issubset(data.columns):
        raise ValueError("CSV must contain Date and Consumption_kWh columns.")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Consumption_kWh"] = pd.to_numeric(data["Consumption_kWh"], errors="coerce")
    data = data.dropna(subset=["Date", "Consumption_kWh"])
    data = data.sort_values("Date").drop_duplicates("Date")
    # Convert readings to daily totals/averages if there are multiple readings per day.
    data = data.set_index("Date").resample("D").mean()
    data["Consumption_kWh"] = data["Consumption_kWh"].interpolate(limit_direction="both")
    if len(data) < 120:
        raise ValueError("At least 120 daily observations are recommended.")
    return data


def make_features(series):
    frame = pd.DataFrame({"y": series})
    for lag in [1, 2, 7, 14, 28]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    frame["rolling_mean_7"] = frame["y"].shift(1).rolling(7).mean()
    frame["rolling_mean_30"] = frame["y"].shift(1).rolling(30).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["day_of_year"] = frame.index.dayofyear
    return frame.dropna()


def train_and_evaluate(data):
    features = make_features(data["Consumption_kWh"])
    split = int(len(features) * 0.80)
    train, test = features.iloc[:split], features.iloc[split:]
    X_train = train.drop(columns="y")
    y_train = train["y"]
    X_test = test.drop(columns="y")
    y_test = test["y"]

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print("\\nMODEL PERFORMANCE (chronological 80/20 split)")
    print(f"MAE : {mae:.2f} kWh")
    print(f"RMSE: {rmse:.2f} kWh")
    print(f"R2  : {r2:.3f}")

    plt.figure(figsize=(12, 5))
    plt.plot(y_test.index, y_test.values, label="Actual")
    plt.plot(y_test.index, predictions, label="Predicted")
    plt.title("Actual vs Predicted Electricity Consumption")
    plt.xlabel("Date")
    plt.ylabel("Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), dpi=150)
    plt.close()

    return model, features, (mae, rmse, r2)


def forecast_future(model, history, days=FORECAST_DAYS):
    values = history["Consumption_kWh"].copy()
    future_dates = pd.date_range(values.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    forecasts = []

    for date in future_dates:
        row = {
            "lag_1": values.iloc[-1],
            "lag_2": values.iloc[-2],
            "lag_7": values.iloc[-7],
            "lag_14": values.iloc[-14],
            "lag_28": values.iloc[-28],
            "rolling_mean_7": values.iloc[-7:].mean(),
            "rolling_mean_30": values.iloc[-30:].mean(),
            "day_of_week": date.dayofweek,
            "month": date.month,
            "day_of_year": date.dayofyear,
        }
        prediction = float(model.predict(pd.DataFrame([row]))[0])
        prediction = max(0.0, prediction)
        forecasts.append(prediction)
        values.loc[date] = prediction

    result = pd.DataFrame({"Date": future_dates, "Forecast_kWh": np.round(forecasts, 2)})
    result.to_csv(os.path.join(OUTPUT_DIR, "next_30_days_forecast.csv"), index=False)

    plt.figure(figsize=(12, 5))
    recent = history["Consumption_kWh"].tail(90)
    plt.plot(recent.index, recent.values, label="Historical (last 90 days)")
    plt.plot(result["Date"], result["Forecast_kWh"], label="30-day forecast")
    plt.title("Electricity Consumption Forecast")
    plt.xlabel("Date")
    plt.ylabel("Consumption (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "future_forecast.png"), dpi=150)
    plt.close()
    return result


def main():
    data = load_data(DATA_FILE)
    print("DATA SUMMARY")
    print(data["Consumption_kWh"].describe().round(2))
    model, features, metrics = train_and_evaluate(data)
    forecast = forecast_future(model, data)
    print("\\nNEXT 30 DAYS")
    print(forecast.to_string(index=False))
    print(f"\\nOutput files saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
