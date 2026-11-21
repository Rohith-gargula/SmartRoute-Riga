
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "reports" / "tables" / "test_predictions.csv"
TABLE_DIR = ROOT / "reports" / "tables"
FIGURE_DIR = ROOT / "reports" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD AND VALIDATE DATA
# --------------------------------------------------

print("Loading test predictions...")

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "forecast_timestamp",
    "transports",
    "marsruts",
    "target_next_hour",
    "persistence_prediction",
    "ml_prediction"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["forecast_timestamp"] = pd.to_datetime(
    df["forecast_timestamp"],
    errors="coerce"
)

numeric_columns = [
    "target_next_hour",
    "persistence_prediction",
    "ml_prediction"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(
    subset=required_columns
).copy()

df["transports"] = df["transports"].astype(str)
df["marsruts"] = df["marsruts"].astype(str)

df["hour"] = df["forecast_timestamp"].dt.hour
df["date"] = df["forecast_timestamp"].dt.date
df["day_of_week"] = df["forecast_timestamp"].dt.day_name()

print(f"Valid prediction rows: {len(df):,}")
print(
    "Test period:",
    df["forecast_timestamp"].min(),
    "to",
    df["forecast_timestamp"].max()
)


# --------------------------------------------------
# 3. CALCULATE PREDICTION ERRORS
# --------------------------------------------------

df["ml_error"] = (
    df["ml_prediction"] - df["target_next_hour"]
)

df["ml_absolute_error"] = df["ml_error"].abs()

df["baseline_error"] = (
    df["persistence_prediction"] - df["target_next_hour"]
)

df["baseline_absolute_error"] = df["baseline_error"].abs()

df["ml_squared_error"] = df["ml_error"] ** 2

df["ml_error_direction"] = np.where(
    df["ml_error"] > 0,
    "Overprediction",
    np.where(
        df["ml_error"] < 0,
        "Underprediction",
        "Exact prediction"
    )
)


# --------------------------------------------------
# 4. METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(group):

    actual = group["target_next_hour"]

    ml_pred = group["ml_prediction"]
    baseline_pred = group["persistence_prediction"]

    actual_total = actual.sum()

    ml_wape = (
        group["ml_absolute_error"].sum() / actual_total * 100
        if actual_total != 0 else np.nan
    )

    baseline_wape = (
        group["baseline_absolute_error"].sum() / actual_total * 100
        if actual_total != 0 else np.nan
    )

    ml_r2 = (
        r2_score(actual, ml_pred)
        if len(group) > 1 and actual.nunique() > 1
        else np.nan
    )

    baseline_r2 = (
        r2_score(actual, baseline_pred)
        if len(group) > 1 and actual.nunique() > 1
        else np.nan
    )

    return pd.Series({
        "observations": len(group),
        "actual_qr_validations": actual_total,
        "average_actual": actual.mean(),

        "ml_MAE": mean_absolute_error(actual, ml_pred),
        "ml_RMSE": np.sqrt(mean_squared_error(actual, ml_pred)),
        "ml_R2": ml_r2,
        "ml_WAPE_percent": ml_wape,
        "ml_bias": (ml_pred - actual).mean(),

        "baseline_MAE": mean_absolute_error(actual, baseline_pred),
        "baseline_RMSE": np.sqrt(
            mean_squared_error(actual, baseline_pred)
        ),
        "baseline_R2": baseline_r2,
        "baseline_WAPE_percent": baseline_wape,

        "MAE_improvement_percent": (
            (
                mean_absolute_error(actual, baseline_pred)
                - mean_absolute_error(actual, ml_pred)
            )
            / mean_absolute_error(actual, baseline_pred)
            * 100
            if mean_absolute_error(actual, baseline_pred) != 0
            else np.nan
        )
    })


# --------------------------------------------------
# 5. OVERALL ERROR SUMMARY
# --------------------------------------------------

print("\nCalculating overall metrics...")

overall = calculate_metrics(df)

overall.to_frame("value").to_csv(
    TABLE_DIR / "overall_error_summary.csv"
)

print("\nOVERALL ERROR ANALYSIS")
print(overall.to_string())


# --------------------------------------------------
# 6. ERROR BY TRANSPORT TYPE
# --------------------------------------------------

print("\nAnalysing transport types...")

transport_results = (
    df.groupby("transports", dropna=False)
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

transport_results.to_csv(
    TABLE_DIR / "error_by_transport.csv",
    index=False
)

print("\nERROR BY TRANSPORT")
print(transport_results.to_string(index=False))


# --------------------------------------------------
# 7. ERROR BY ROUTE
# --------------------------------------------------

print("\nAnalysing individual routes...")

route_results = (
    df.groupby(["transports", "marsruts"], dropna=False)
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

route_results = route_results.sort_values(
    "observations",
    ascending=False
)

route_results.to_csv(
    TABLE_DIR / "error_by_route.csv",
    index=False
)

print("\nROUTE RESULTS - 20 ROUTES WITH MOST TEST OBSERVATIONS")
print(route_results.head(20).to_string(index=False))


# --------------------------------------------------
# 8. ERROR BY HOUR OF DAY
# --------------------------------------------------

print("\nAnalysing hourly performance...")

hour_results = (
    df.groupby("hour")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
    .sort_values("hour")
)

hour_results.to_csv(
    TABLE_DIR / "error_by_hour.csv",
    index=False
)

print("\nERROR BY HOUR")
print(hour_results.to_string(index=False))


# --------------------------------------------------
# 9. PEAK AND OFF-PEAK ANALYSIS
# --------------------------------------------------

# Operational definition used for this experiment:
# Morning peak: 07:00-09:59
# Evening peak: 16:00-18:59
# All other hours: Off-peak
#
# These are analytical time bands, not verified
# official Riga public transport peak-hour definitions.

peak_hours = [7, 8, 9, 16, 17, 18]

df["period_type"] = np.where(
    df["hour"].isin(peak_hours),
    "Peak",
    "Off-peak"
)

peak_results = (
    df.groupby("period_type")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

peak_results.to_csv(
    TABLE_DIR / "error_peak_vs_offpeak.csv",
    index=False
)

print("\nPEAK VS OFF-PEAK RESULTS")
print(peak_results.to_string(index=False))


# --------------------------------------------------
# 10. LARGEST PREDICTION ERRORS
# --------------------------------------------------

largest_errors = df.sort_values(
    "ml_absolute_error",
    ascending=False
).head(30).copy()

largest_errors = largest_errors[
    [
        "forecast_timestamp",
        "transports",
        "marsruts",
        "target_next_hour",
        "ml_prediction",
        "persistence_prediction",
        "ml_error",
        "ml_absolute_error",
        "ml_error_direction"
    ]
]

largest_errors.to_csv(
    TABLE_DIR / "largest_prediction_errors.csv",
    index=False
)

print("\n30 LARGEST RANDOM FOREST ERRORS")
print(largest_errors.to_string(index=False))


# --------------------------------------------------
# 11. ERROR DIRECTION SUMMARY
# --------------------------------------------------

direction_summary = (
    df["ml_error_direction"]
    .value_counts()
    .rename_axis("error_direction")
    .reset_index(name="observations")
)

direction_summary["percentage"] = (
    direction_summary["observations"]
    / len(df) * 100
)

direction_summary.to_csv(
    TABLE_DIR / "prediction_error_direction.csv",
    index=False
)

print("\nPREDICTION ERROR DIRECTION")
print(direction_summary.to_string(index=False))


# --------------------------------------------------
# 12. CHART 1 - ACTUAL VS PREDICTED
# --------------------------------------------------

print("\nCreating charts...")

sample = df.sample(
    n=min(10000, len(df)),
    random_state=42
)

plt.figure(figsize=(9, 7))

plt.scatter(
    sample["target_next_hour"],
    sample["ml_prediction"],
    alpha=0.25
)

maximum = max(
    sample["target_next_hour"].max(),
    sample["ml_prediction"].max()
)

plt.plot(
    [0, maximum],
    [0, maximum],
    linestyle="--"
)

plt.xlabel("Actual next-hour QR validations")
plt.ylabel("Random Forest predicted QR validations")
plt.title("Actual vs Predicted QR Validations - Test Data")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 13. CHART 2 - MAE BY HOUR
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    hour_results["hour"],
    hour_results["ml_MAE"]
)

plt.xlabel("Forecast hour")
plt.ylabel("Mean Absolute Error")
plt.title("Random Forest MAE by Hour of Day")
plt.xticks(range(24))
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "mae_by_hour.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. CHART 3 - MAE BY TRANSPORT
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    transport_results["transports"],
    transport_results["ml_MAE"]
)

plt.xlabel("Transport type")
plt.ylabel("Mean Absolute Error")
plt.title("Random Forest MAE by Transport Type")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "mae_by_transport.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 15. CHART 4 - ERROR DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.hist(
    df["ml_error"],
    bins=60
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel("Prediction error (predicted - actual)")
plt.ylabel("Number of observations")
plt.title("Random Forest Prediction Error Distribution")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "prediction_error_distribution.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 16. COMPLETION
# --------------------------------------------------

print("\nMODEL ERROR ANALYSIS COMPLETED.")

print("\nTABLES SAVED:")
print(" - overall_error_summary.csv")
print(" - error_by_transport.csv")
print(" - error_by_route.csv")
print(" - error_by_hour.csv")
print(" - error_peak_vs_offpeak.csv")
print(" - largest_prediction_errors.csv")
print(" - prediction_error_direction.csv")

print("\nFIGURES SAVED:")
print(" - actual_vs_predicted.png")
print(" - mae_by_hour.png")
print(" - mae_by_transport.png")
print(" - prediction_error_distribution.png")

print("\nNo model retraining was performed.")
