
import os
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(".")
DATA_FILE = ROOT / "data" / "processed" / "next_hour_model_dataset.csv"
TABLE_DIR = ROOT / "reports" / "tables"
MODEL_DIR = ROOT / "models"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("\nLoading forecasting dataset...")
df = pd.read_csv(DATA_FILE, low_memory=False)

df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])
df["transports"] = df["transports"].astype(str)
df["marsruts"] = df["marsruts"].astype(str)
df["target_next_hour"] = pd.to_numeric(
    df["target_next_hour"], errors="coerce"
)

df = df.dropna(
    subset=["forecast_timestamp", "target_next_hour"]
).copy()

print("Dataset rows:", len(df))
print("Date range:", df["forecast_timestamp"].min(),
      "to", df["forecast_timestamp"].max())

train = df[df["forecast_timestamp"] < "2026-07-01"].copy()

validation = df[
    (df["forecast_timestamp"] >= "2026-07-01") &
    (df["forecast_timestamp"] < "2026-09-01")
].copy()

test = df[
    (df["forecast_timestamp"] >= "2026-09-01") &
    (df["forecast_timestamp"] < "2026-11-01")
].copy()

print("\nCHRONOLOGICAL DATA SPLIT")
print("Training rows:", len(train))
print("Validation rows:", len(validation))
print("Testing rows:", len(test))

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("One of the chronological data splits is empty.")

categorical_features = ["transports", "marsruts"]

numeric_features = [
    "skaits",
    "qr_validations_previous_hour",
    "qr_validations_same_hour_previous_day",
    "qr_validations_same_hour_previous_week",
    "forecast_hour",
    "forecast_day_of_week",
    "forecast_month",
    "forecast_day",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos"
]

feature_columns = categorical_features + numeric_features
target_column = "target_next_hour"

X_train = train[feature_columns]
y_train = train[target_column]

X_validation = validation[feature_columns]
y_validation = validation[target_column]

X_test = test[feature_columns]
y_test = test[target_column]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True))
])

categorical_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    wape = (
        np.sum(np.abs(actual - predicted)) /
        np.sum(np.abs(actual)) * 100
    )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "WAPE_percent": wape
    }

print("\nEvaluating persistence baseline...")

baseline_validation = evaluate_model(
    "Persistence baseline",
    y_validation,
    validation["skaits"]
)

print(baseline_validation)


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=30,
        max_depth=18,
        max_features=0.5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=30,
        max_depth=18,
        max_features=0.5,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}
validation_results = [baseline_validation]

for name, estimator in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_seconds = time.time() - start_time

    predictions = pipeline.predict(X_validation)

    result = evaluate_model(name, y_validation, predictions)
    result["training_seconds"] = training_seconds
    validation_results.append(result)

    print("Validation results:", result)
    print("Training time:", round(training_seconds, 2), "seconds")

validation_table = pd.DataFrame(validation_results)
validation_table = validation_table.sort_values("MAE")

validation_path = TABLE_DIR / "model_comparison_validation.csv"
validation_table.to_csv(validation_path, index=False)

print("\nVALIDATION COMPARISON")
print(validation_table.to_string(index=False))

ml_results = validation_table[
    validation_table["model"].isin(models.keys())
]

best_name = ml_results.iloc[0]["model"]
print("\nSelected ML model:", best_name)

development_data = pd.concat(
    [train, validation], ignore_index=True
)

X_development = development_data[feature_columns]
y_development = development_data[target_column]

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", models[best_name])
])

print("\nRefitting selected model on training + validation...")
start_time = time.time()

final_model.fit(X_development, y_development)

final_training_seconds = time.time() - start_time

print("\nEvaluating on untouched test data...")
test_predictions = final_model.predict(X_test)

baseline_test = evaluate_model(
    "Persistence baseline", y_test, test["skaits"]
)

ml_test = evaluate_model(
    best_name, y_test, test_predictions
)

test_results = pd.DataFrame([baseline_test, ml_test])
test_results["final_training_seconds"] = [
    np.nan, final_training_seconds
]

test_results.to_csv(
    TABLE_DIR / "final_test_metrics.csv", index=False
)

print("\nFINAL TEST RESULTS")
print(test_results.to_string(index=False))

prediction_output = test[
    ["forecast_timestamp", "transports", "marsruts",
     "target_next_hour", "skaits"]
].copy()

prediction_output["persistence_prediction"] = test["skaits"].to_numpy()
prediction_output["ml_prediction"] = test_predictions

prediction_output.to_csv(
    TABLE_DIR / "test_predictions.csv", index=False
)

print("\nCalculating feature importance...")

feature_names = (
    final_model.named_steps["preprocessor"].get_feature_names_out()
)

importance_values = (
    final_model.named_steps["model"].feature_importances_
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance["feature"] = (
    importance["feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
)

importance = importance.sort_values("importance", ascending=False)

importance.to_csv(
    TABLE_DIR / "feature_importance.csv", index=False
)

print("\nTOP 20 FEATURES")
print(importance.head(20).to_string(index=False))

model_path = MODEL_DIR / "best_next_hour_model.joblib"

joblib.dump(final_model, model_path, compress=3)

model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

print("\nFINAL MODEL SAVED")
print("Model:", model_path)
print("Model size:", round(model_size_mb, 2), "MB")
print("Final training time:", round(final_training_seconds, 2), "seconds")

print("\nMODEL TRAINING AND EVALUATION COMPLETED.")
