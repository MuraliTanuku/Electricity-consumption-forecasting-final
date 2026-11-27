from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from model_engine import run_experiment

BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"
app = Flask(__name__, static_folder="static")

@app.get("/")
def home():
    return send_from_directory(BASE / "static", "index.html")

@app.post("/api/run")
def run():
    try:
        uploaded = request.files.get("file")
        source = uploaded.stream if uploaded and uploaded.filename else BASE / "electricity_consumption.csv"
        result = run_experiment(source, OUTPUTS, forecast_days=30)
        return jsonify({"ok": True, **result})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

@app.get("/api/download/<name>")
def download(name):
    allowed = {"model_comparison.csv", "future_forecast.csv"}
    if name not in allowed:
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(OUTPUTS, name, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
