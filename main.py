
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from prefect import flow, task


# 1. CONFIGURATION
FEATURES = ["temperature", "promotion", "day_of_week"]
TARGET = "demand"

DRIFT_THRESHOLD = 0.1
MODEL_NAME = "DemandForecastModel"

os.makedirs("reports", exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Demand_Forecasting")


# 2. GENERATE SYNTHETIC DATA
@task
def generate_data():
    np.random.seed(42)
    n = 1000

    reference = pd.DataFrame({
        "temperature": np.random.normal(25, 5, n),
        "promotion": np.random.randint(0, 2, n),
        "day_of_week": np.random.randint(0, 7, n)
    })

    reference["demand"] = (
        100
        + 2 * reference["temperature"]
        + 25 * reference["promotion"]
        - 3 * reference["day_of_week"]
        + np.random.normal(0, 5, n)
    )

    # Simulate incoming data with distribution drift
    current = pd.DataFrame({
        "temperature": np.random.normal(32, 5, n),
        "promotion": np.random.randint(0, 2, n),
        "day_of_week": np.random.randint(0, 7, n)
    })

    current["demand"] = (
        100
        + 2 * current["temperature"]
        + 25 * current["promotion"]
        - 3 * current["day_of_week"]
        + np.random.normal(0, 5, n)
    )

    reference.to_csv("reference.csv", index=False)
    current.to_csv("current.csv", index=False)

    print("Reference and current data generated.")
    return reference, current


# 3. MONITOR DATA DRIFT
@task
def detect_drift(reference, current):
    # Generate an Evidently drift report
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[FEATURES],
        current_data=current[FEATURES]
    )
    report.save_html("reports/drift_report.html")

    # Calculate normalized Wasserstein distance
    distances = {}

    for col in FEATURES:
        distance = wasserstein_distance(
            reference[col], current[col]
        )

        scale = reference[col].std()
        distances[col] = distance / max(scale, 1e-8)

    max_drift = max(distances.values())

    print("\nDrift scores:", distances)
    print("Maximum drift:", round(max_drift, 4))
    print("Threshold:", DRIFT_THRESHOLD)

    if max_drift > DRIFT_THRESHOLD:
        print("DRIFT DETECTED")
        return True

    print("No significant drift detected")
    return False


# 4. RETRAIN MODEL AND REGISTER IN MLFLOW
@task
def retrain_model(current):
    X = current[FEATURES]
    y = current[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print("Retrained model MAE:", round(mae, 4))

    with mlflow.start_run() as run:
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("MAE", mae)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        print("MLflow Run ID:", run.info.run_id)
        print("Model registered:", MODEL_NAME)

    return mae


# 5. SEND ALERT
@task
def send_alert(message):
    print("\n========== ALERT ==========")
    print(message)
    print("===========================\n")

    # Optional: configure a Discord webhook.
    webhook = os.getenv("DISCORD_WEBHOOK_URL")

    if webhook:
        import requests
        response = requests.post(
            webhook,
            json={"content": message},
            timeout=10
        )
        response.raise_for_status()


# 6. AUTOMATED PIPELINE
@flow(name="Automated-Drift-Retraining-Pipeline")
def drift_retraining_pipeline():
    reference, current = generate_data()

    drift_detected = detect_drift(reference, current)

    if drift_detected:
        mae = retrain_model(current)
        send_alert(
            f"Data drift detected. Model retrained and "
            f"registered. Test MAE: {mae:.4f}"
        )
    else:
        send_alert("No significant drift. Retraining skipped.")


if __name__ == "__main__":
    drift_retraining_pipeline()
