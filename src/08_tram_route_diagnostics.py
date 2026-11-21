
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = ROOT / "reports" / "tables" / "test_predictions.csv"
OUTPUT = ROOT / "reports" / "tables"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS, dtype={"marsruts": str})
df["forecast_timestamp"] = pd.to_datetime(df["forecast_timestamp"])

# Isolate Tramvajs route 1 at 17:00.
route = df[
    (df["transports"] == "Tramvajs")
    & (df["marsruts"].str.strip() == "1")
    & (df["forecast_timestamp"].dt.hour == 17)
].copy()

route["weekday"] = route["forecast_timestamp"].dt.day_name()
route["rf_error"] = route["ml_prediction"] - route["target_next_hour"]
route["baseline_error"] = (
    route["persistence_prediction"] - route["target_next_hour"]
)

# Weekday comparison.
weekday_results = (
    route.groupby("weekday")
    .apply(
        lambda g: pd.Series({
            "observations": len(g),
            "actual_mean": g["target_next_hour"].mean(),
            "RF_MAE": g["rf_error"].abs().mean(),
            "Baseline_MAE": g["baseline_error"].abs().mean(),
            "RF_bias": g["rf_error"].mean(),
            "Baseline_bias": g["baseline_error"].mean(),
        }),
        include_groups=False
    )
    .reset_index()
)

weekday_results.to_csv(
    OUTPUT / "tram_route1_17_weekday_comparison.csv",
    index=False
)

# Largest prediction errors.
largest_errors = route.assign(
    absolute_error=route["rf_error"].abs()
).sort_values("absolute_error", ascending=False)

largest_errors[
    [
        "forecast_timestamp",
        "weekday",
        "target_next_hour",
        "ml_prediction",
        "persistence_prediction",
        "rf_error",
        "absolute_error",
    ]
].head(20).to_csv(
    OUTPUT / "tram_route1_17_largest_errors.csv",
    index=False
)

# Overall comparison.
overall = pd.DataFrame([{
    "transport": "Tramvajs",
    "route": "1",
    "forecast_hour": 17,
    "observations": len(route),
    "actual_total": route["target_next_hour"].sum(),
    "RF_MAE": route["rf_error"].abs().mean(),
    "Baseline_MAE": route["baseline_error"].abs().mean(),
    "RF_WAPE_percent": (
        route["rf_error"].abs().sum()
        / route["target_next_hour"].sum() * 100
    ),
    "Baseline_WAPE_percent": (
        route["baseline_error"].abs().sum()
        / route["target_next_hour"].sum() * 100
    ),
    "RF_bias": route["rf_error"].mean(),
    "Baseline_bias": route["baseline_error"].mean(),
}])

overall.to_csv(
    OUTPUT / "tram_route1_17_overall_comparison.csv",
    index=False
)

print("\nTRAM ROUTE 1 DIAGNOSTICS SAVED")
print("Observations:", len(route))
print("\nOverall comparison:")
print(overall.round(2).to_string(index=False))
print("\nWeekday comparison:")
print(weekday_results.round(2).to_string(index=False))
print("\nCreated files:")
print("tram_route1_17_weekday_comparison.csv")
print("tram_route1_17_largest_errors.csv")
print("tram_route1_17_overall_comparison.csv")
