"""Command-line entry point for the electricity forecasting project."""
from pathlib import Path
from model_engine import run_experiment

if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    result = run_experiment(base / "electricity_consumption.csv", base / "outputs", 30)
    print("MODEL COMPARISON (chronological 80/20 split)")
    for row in result["metrics"]:
        print(row)
    print("Best model:", result["summary"]["best_model"])
    print("30-day forecast saved in outputs/future_forecast.csv")
