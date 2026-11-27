"""Model comparison and forecasting engine for electricity consumption."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_30", "day_of_week", "month", "day_of_year"
]


def load_data(source):
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    required = {"Date", "Consumption_kWh"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain exactly the required fields: Date and Consumption_kWh.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Consumption_kWh"] = pd.to_numeric(df["Consumption_kWh"], errors="coerce")
    df = df.dropna(subset=["Date", "Consumption_kWh"])
    df = df[df["Consumption_kWh"] >= 0].sort_values("Date")
    if df.empty:
        raise ValueError("No valid, non-negative consumption observations were found.")
    # Daily mean matches the aggregation approach in the supplied starter project.
    daily = df.set_index("Date")["Consumption_kWh"].resample("D").mean()
    daily = daily.interpolate(limit_direction="both").dropna()
    if len(daily) < 120:
        raise ValueError(f"At least 120 daily observations are recommended; found {len(daily)}.")
    return daily


def make_features(series):
    frame = pd.DataFrame({"target": series})
    for lag in [1, 2, 7, 14, 28]:
        frame[f"lag_{lag}"] = series.shift(lag)
    frame["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    frame["rolling_mean_30"] = series.shift(1).rolling(30).mean()
    frame["day_of_week"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["day_of_year"] = frame.index.dayofyear
    return frame.dropna()


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(actual, predicted) if len(actual) > 1 else float("nan")
    nonzero = np.abs(actual) > 1e-9
    mape = float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "R2": float(r2), "MAPE": mape}


def recursive_forecast(model, history, days=30):
    values = history.copy().astype(float)
    rows = []
    dates = pd.date_range(values.index[-1] + pd.Timedelta(days=1), periods=days, freq="D")
    for date in dates:
        row = {
            "lag_1": values.iloc[-1], "lag_2": values.iloc[-2],
            "lag_7": values.iloc[-7], "lag_14": values.iloc[-14],
            "lag_28": values.iloc[-28],
            "rolling_mean_7": values.iloc[-7:].mean(),
            "rolling_mean_30": values.iloc[-30:].mean(),
            "day_of_week": date.dayofweek, "month": date.month,
            "day_of_year": date.dayofyear,
        }
        prediction = max(0.0, float(model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
        values.loc[date] = prediction
        rows.append({"Date": date.strftime("%Y-%m-%d"), "Forecast_kWh": round(prediction, 3)})
    return rows


def run_experiment(source, output_dir="outputs", forecast_days=30):
    daily = load_data(source)
    frame = make_features(daily)
    split = int(len(frame) * 0.80)
    if split < 30 or len(frame) - split < 10:
        raise ValueError("Not enough rows after feature creation for a reliable 80/20 chronological split.")
    train, test = frame.iloc[:split], frame.iloc[split:]
    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, loss="huber", random_state=42),
    }
    results, fitted = [], {}
    # Persistence baseline: tomorrow is predicted to equal the latest observed day.
    predictions = {"Naive (last day)": X_test["lag_1"].to_numpy()}
    for name, model in candidates.items():
        fitted[name] = clone(model).fit(X_train, y_train)
        predictions[name] = fitted[name].predict(X_test)

    # Tune Random Forest only on the training period using expanding time-series folds.
    tuned = RandomForestRegressor(random_state=42, n_jobs=-1)
    cv = TimeSeriesSplit(n_splits=3)
    search = GridSearchCV(
        tuned,
        {"n_estimators": [150, 300], "max_depth": [8, 12, None], "min_samples_leaf": [1, 2]},
        scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    fitted["Tuned Random Forest"] = search.best_estimator_
    predictions["Tuned Random Forest"] = search.predict(X_test)

    for name, pred in predictions.items():
        metrics = calculate_metrics(y_test, pred)
        results.append({"Model": name, **metrics})
    results.sort(key=lambda item: item["RMSE"])
    best_name = results[0]["Model"]
    best_model = fitted[best_name] if best_name in fitted else None
    if best_model is None:
        # The naive baseline has no fitted estimator; use a one-step persistence forecaster.
        class PersistenceModel:
            def predict(self, X):
                return X["lag_1"].to_numpy()
        best_model = PersistenceModel()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "model_comparison.csv", index=False)
    forecast = recursive_forecast(best_model, daily, forecast_days)
    pd.DataFrame(forecast).to_csv(out / "future_forecast.csv", index=False)
    comparison_actual = [
        {"Date": idx.strftime("%Y-%m-%d"), "Actual": round(float(actual), 3),
         "Predicted": round(float(predictions[best_name][i]), 3)}
        for i, (idx, actual) in enumerate(zip(y_test.index, y_test.to_numpy()))
    ]
    summary = {
        "observations": int(len(daily)),
        "start_date": daily.index.min().strftime("%Y-%m-%d"),
        "end_date": daily.index.max().strftime("%Y-%m-%d"),
        "mean_kwh": round(float(daily.mean()), 3),
        "best_model": best_name,
        "best_parameters": search.best_params_ if best_name == "Tuned Random Forest" else {},
        "train_rows": int(len(train)), "test_rows": int(len(test)),
    }
    return {"summary": summary, "metrics": results, "forecast": forecast,
            "test_predictions": comparison_actual, "historical": [
                {"Date": idx.strftime("%Y-%m-%d"), "Consumption_kWh": round(float(val), 3)}
                for idx, val in daily.tail(120).items()
            ]}
