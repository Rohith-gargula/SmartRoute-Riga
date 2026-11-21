
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "combined_qr_validations.csv"
TABLE_DIR = ROOT / "reports" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SMARTROUTE-RIGA: DATA COMPLETENESS AUDIT")
print("=" * 65)

# Check input file
if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE, low_memory=False)

# Standardise column names
df.columns = [str(col).strip().lower() for col in df.columns]

required = ["datums", "stunda", "transports", "marsruts", "skaits"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse dates and values
df["datums"] = pd.to_datetime(df["datums"], errors="coerce")
df["skaits"] = pd.to_numeric(df["skaits"], errors="coerce")

df["transports"] = df["transports"].astype("string").str.strip()
df["marsruts"] = df["marsruts"].astype("string").str.strip()
df["stunda"] = df["stunda"].astype("string").str.strip()

df = df.dropna(subset=["datums", "skaits", "transports", "marsruts", "stunda"])

df["date"] = df["datums"].dt.date
df["year_month"] = df["datums"].dt.to_period("M").astype(str)

# 1. Date coverage
first_date = df["datums"].min()
last_date = df["datums"].max()

expected_dates = pd.date_range(first_date, last_date, freq="D")
observed_dates = pd.DatetimeIndex(df["datums"].dt.normalize().unique())

missing_dates = expected_dates.difference(observed_dates)

print("\n1. DATE COVERAGE")
print(f"First date: {first_date.date()}")
print(f"Last date: {last_date.date()}")
print(f"Calendar days in range: {len(expected_dates)}")
print(f"Dates with records: {len(observed_dates)}")
print(f"Dates without records: {len(missing_dates)}")

if len(missing_dates) > 0:
    print("\nDates without records:")
    for date in missing_dates:
        print(date.date())

missing_dates_df = pd.DataFrame({
    "missing_date": missing_dates.date
})
missing_dates_df.to_csv(
    TABLE_DIR / "missing_dates.csv", index=False
)

# 2. Daily record coverage
daily = (
    df.groupby("date")
    .agg(
        records=("skaits", "size"),
        total_qr_validations=("skaits", "sum"),
        routes=("marsruts", "nunique"),
        transport_types=("transports", "nunique")
    )
    .reset_index()
)

daily.to_csv(TABLE_DIR / "daily_completeness.csv", index=False)

print("\n2. DAILY RECORD COVERAGE")
print(daily["records"].describe().round(2).to_string())

# 3. Hourly interval coverage
interval_summary = (
    df.groupby("stunda")
    .agg(
        records=("skaits", "size"),
        total_qr_validations=("skaits", "sum"),
        unique_dates=("date", "nunique")
    )
    .reset_index()
    .sort_values("stunda")
)

interval_summary.to_csv(
    TABLE_DIR / "hourly_interval_coverage.csv", index=False
)

print("\n3. HOURLY INTERVAL COVERAGE")
print(interval_summary.to_string(index=False))

# 4. Transport coverage
transport_summary = (
    df.groupby("transports")
    .agg(
        records=("skaits", "size"),
        total_qr_validations=("skaits", "sum"),
        unique_routes=("marsruts", "nunique"),
        unique_dates=("date", "nunique")
    )
    .reset_index()
    .sort_values("total_qr_validations", ascending=False)
)

transport_summary.to_csv(
    TABLE_DIR / "transport_completeness.csv", index=False
)

print("\n4. TRANSPORT COVERAGE")
print(transport_summary.to_string(index=False))

# 5. Route-level date coverage
route_summary = (
    df.groupby(["transports", "marsruts"])
    .agg(
        records=("skaits", "size"),
        total_qr_validations=("skaits", "sum"),
        unique_dates=("date", "nunique"),
        unique_intervals=("stunda", "nunique")
    )
    .reset_index()
    .sort_values(
        ["transports", "total_qr_validations"],
        ascending=[True, False]
    )
)

route_summary.to_csv(
    TABLE_DIR / "route_completeness.csv", index=False
)

print("\n5. ROUTE COVERAGE")
print(f"Transport-route combinations: {len(route_summary)}")
print(route_summary.head(15).to_string(index=False))

# 6. Duplicate key check
key_columns = ["datums", "stunda", "transports", "marsruts"]
duplicate_count = int(df.duplicated(subset=key_columns).sum())

print("\n6. DUPLICATE CHECK")
print(f"Duplicate date-hour-transport-route records: {duplicate_count}")

# 7. Missing-value and count checks
print("\n7. VALIDATION CHECKS")
print("Missing values:")
print(df[required].isna().sum().to_string())

print(f"\nZero or negative validation counts: {(df['skaits'] <= 0).sum()}")
print(f"Rows after valid date/count/category filtering: {len(df)}")

# Important interpretation note
print("\n8. IMPORTANT DATA LIMITATION")
print(
    "The dataset records QR ticket validations, not all passengers. "
    "Dates or route-hour combinations without records must not "
    "automatically be interpreted as zero passenger demand."
)

print("\nAUDIT COMPLETED")
print(f"Output folder: {TABLE_DIR}")
print("=" * 65)
