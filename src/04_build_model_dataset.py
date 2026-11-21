
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# SMARTROUTE-RIGA: BUILD NEXT-HOUR FORECASTING DATASET
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "combined_qr_validations.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("SMARTROUTE-RIGA: NEXT-HOUR MODEL DATASET PREPARATION")
print("=" * 70)

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

# 1. Load data
df = pd.read_csv(INPUT_FILE, low_memory=False)
df.columns = [str(c).strip().lower() for c in df.columns]

required = ["datums", "stunda", "transports", "marsruts", "skaits"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df["datums"] = pd.to_datetime(df["datums"], errors="coerce")
df["skaits"] = pd.to_numeric(df["skaits"], errors="coerce")

for col in ["stunda", "transports", "marsruts"]:
    df[col] = df[col].astype("string").str.strip()

df = df.dropna(subset=required).copy()

# 2. Extract the starting hour from intervals such as 08:00-09:00
df["start_hour"] = pd.to_numeric(
    df["stunda"].str.extract(r"^(\d{1,2}):")[0],
    errors="coerce"
)

df = df.dropna(subset=["start_hour"]).copy()
df["start_hour"] = df["start_hour"].astype(int)

if not df["start_hour"].between(0, 23).all():
    raise ValueError("Unexpected hour values found.")

# Create a local calendar timestamp.
# The source does not provide timezone-aware timestamps.
df["timestamp"] = (
    df["datums"].dt.normalize()
    + pd.to_timedelta(df["start_hour"], unit="h")
)

route_keys = ["transports", "marsruts"]
key_columns = route_keys + ["timestamp"]

# Each route should have no more than one record per time interval.
duplicates = int(df.duplicated(subset=key_columns).sum())

if duplicates:
    raise ValueError(
        f"Found {duplicates} duplicate route-hour records. "
        "Resolve duplicates before modelling."
    )

base = df[
    route_keys + ["timestamp", "skaits"]
].copy()

base = base.sort_values(key_columns).reset_index(drop=True)

print(f"\nSource records: {len(base):,}")
print(f"Transport-route combinations: {base.groupby(route_keys).ngroups:,}")

# 3. Create the next-hour prediction target.
# Shift each observed record back one hour so it aligns with
# the preceding hour's feature row.
target_lookup = base[
    route_keys + ["timestamp", "skaits"]
].copy()

target_lookup["timestamp"] = (
    target_lookup["timestamp"] - pd.Timedelta(hours=1)
)

target_lookup = target_lookup.rename(
    columns={"skaits": "target_next_hour"}
)

model_df = base.merge(
    target_lookup,
    on=key_columns,
    how="left",
    validate="one_to_one"
)

# Keep only rows where the actual next-hour observation exists.
model_df = model_df.dropna(subset=["target_next_hour"]).copy()

# 4. Create historical lag features.
# Each lag is matched by its exact timestamp and same route.
def add_lag_feature(data, lag_hours, feature_name):
    lookup = base[
        route_keys + ["timestamp", "skaits"]
    ].copy()

    # Move historical observation forward to its prediction row.
    lookup["timestamp"] = (
        lookup["timestamp"] + pd.Timedelta(hours=lag_hours)
    )

    lookup = lookup.rename(
        columns={"skaits": feature_name}
    )

    return data.merge(
        lookup,
        on=key_columns,
        how="left",
        validate="one_to_one"
    )

model_df = add_lag_feature(
    model_df, 1, "qr_validations_previous_hour"
)

model_df = add_lag_feature(
    model_df, 24, "qr_validations_same_hour_previous_day"
)

model_df = add_lag_feature(
    model_df, 168, "qr_validations_same_hour_previous_week"
)

# 5. Create features known at the forecast time.
model_df["forecast_timestamp"] = (
    model_df["timestamp"] + pd.Timedelta(hours=1)
)

model_df["forecast_hour"] = model_df["forecast_timestamp"].dt.hour
model_df["forecast_day_of_week"] = (
    model_df["forecast_timestamp"].dt.dayofweek
)
model_df["forecast_month"] = model_df["forecast_timestamp"].dt.month
model_df["forecast_day"] = model_df["forecast_timestamp"].dt.day

model_df["is_weekend"] = (
    model_df["forecast_day_of_week"] >= 5
).astype(int)

model_df["hour_sin"] = np.sin(
    2 * np.pi * model_df["forecast_hour"] / 24
)

model_df["hour_cos"] = np.cos(
    2 * np.pi * model_df["forecast_hour"] / 24
)

model_df["day_sin"] = np.sin(
    2 * np.pi * model_df["forecast_day_of_week"] / 7
)

model_df["day_cos"] = np.cos(
    2 * np.pi * model_df["forecast_day_of_week"] / 7
)

# 6. Select and order final columns.
final_columns = [
    "timestamp",
    "forecast_timestamp",
    "transports",
    "marsruts",
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
    "day_cos",
    "target_next_hour"
]

model_df = model_df[final_columns].copy()

model_df = model_df.sort_values(
    ["forecast_timestamp", "transports", "marsruts"]
).reset_index(drop=True)

# 7. Save the modelling dataset.
output_file = OUTPUT_DIR / "next_hour_model_dataset.csv"
model_df.to_csv(output_file, index=False)

# 8. Generate a dataset summary.
summary = pd.DataFrame({
    "metric": [
        "Source records",
        "Model dataset rows",
        "Transport-route combinations",
        "First forecast timestamp",
        "Last forecast timestamp",
        "Target missing values",
        "Current QR validation missing values",
        "Previous-hour feature missing values",
        "Previous-day feature missing values",
        "Previous-week feature missing values"
    ],
    "value": [
        len(base),
        len(model_df),
        model_df.groupby(route_keys).ngroups,
        str(model_df["forecast_timestamp"].min()),
        str(model_df["forecast_timestamp"].max()),
        int(model_df["target_next_hour"].isna().sum()),
        int(model_df["skaits"].isna().sum()),
        int(model_df["qr_validations_previous_hour"].isna().sum()),
        int(model_df["qr_validations_same_hour_previous_day"].isna().sum()),
        int(model_df["qr_validations_same_hour_previous_week"].isna().sum())
    ]
})

summary.to_csv(
    REPORT_DIR / "model_dataset_summary.csv",
    index=False
)

# 9. Report missingness in model features.
missingness = (
    model_df.isna().sum()
    .rename("missing_count")
    .to_frame()
    .reset_index()
    .rename(columns={"index": "column"})
)

missingness["missing_percentage"] = (
    missingness["missing_count"] / len(model_df) * 100
).round(2)

missingness.to_csv(
    REPORT_DIR / "model_feature_missingness.csv",
    index=False
)

# 10. Display results.
print("\nMODEL DATASET SUMMARY")
print(summary.to_string(index=False))

print("\nFEATURE MISSINGNESS")
print(missingness.to_string(index=False))

print("\nFIRST FIVE ROWS")
print(model_df.head().to_string(index=False))

print("\nOUTPUT FILES")
print(f"Dataset: {output_file}")
print(f"Summary: {REPORT_DIR / 'model_dataset_summary.csv'}")
print(f"Missingness: {REPORT_DIR / 'model_feature_missingness.csv'}")

print("\nIMPORTANT:")
print(
    "The target represents next-hour QR ticket validations, "
    "not total passenger demand. Missing historical observations "
    "are preserved as missing values, not converted to zero."
)

print("\nDATASET PREPARATION COMPLETED")
print("=" * 70)
