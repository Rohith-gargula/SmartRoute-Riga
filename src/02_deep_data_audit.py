
from pathlib import Path
import pandas as pd
import re

# ==========================================
# SMARTROUTE-RIGA: DEEP DATA AUDIT
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "combined_qr_validations.csv"
REPORT_DIR = PROJECT_DIR / "reports" / "tables"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

# Normalise route values without losing route labels
def normalise_route(value):
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    return value

df["route_clean"] = df["marsruts"].apply(normalise_route)
df["transport_clean"] = df["transports"].astype(str).str.strip()

df["datums"] = pd.to_datetime(
    df["datums"], errors="coerce"
)

# ==========================================
# 1. ROUTE ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("ROUTE ANALYSIS")
print("=" * 60)

print("\nDistinct routes overall:", df["route_clean"].nunique())

print("\nDistinct routes by transport:")
print(
    df.groupby("transport_clean")["route_clean"]
    .nunique()
)

print("\nTop 20 transport-route combinations:")
route_summary = (
    df.groupby(["transport_clean", "route_clean"])["skaits"]
    .agg(["sum", "count", "mean"])
    .sort_values("sum", ascending=False)
    .head(20)
)

print(route_summary.to_string())

route_summary.to_csv(
    REPORT_DIR / "route_summary.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 2. TIME INTERVAL ANALYSIS
# ==========================================

print("\n" + "=" * 60)
print("TIME INTERVAL ANALYSIS")
print("=" * 60)

intervals = df["stunda"].astype(str).str.strip()

pattern = r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$"
parts = intervals.str.extract(pattern)

df["start_hour"] = pd.to_numeric(parts[0], errors="coerce")
df["start_minute"] = pd.to_numeric(parts[1], errors="coerce")
df["end_hour"] = pd.to_numeric(parts[2], errors="coerce")
df["end_minute"] = pd.to_numeric(parts[3], errors="coerce")

valid_time = parts.notna().all(axis=1)

start_minutes = (
    df["start_hour"] * 60 + df["start_minute"]
)

end_minutes = (
    df["end_hour"] * 60 + df["end_minute"]
)

duration = end_minutes - start_minutes
duration = duration.where(duration > 0, duration + 1440)

df["interval_minutes"] = duration

print("\nUnique time intervals:", intervals.nunique())
print("\nMost common intervals:")
print(intervals.value_counts().head(15))

print("\nRows with invalid interval format:", (~valid_time).sum())
print("\nInterval duration distribution:")
print(df["interval_minutes"].describe())

# ==========================================
# 3. CHECK DUPLICATE TIME-SERIES KEYS
# ==========================================

print("\n" + "=" * 60)
print("TIME-SERIES KEY CHECK")
print("=" * 60)

key_columns = [
    "datums",
    "stunda",
    "transport_clean",
    "route_clean"
]

duplicate_keys = df.duplicated(
    subset=key_columns,
    keep=False
)

print("\nRows involved in repeated route-time keys:",
      duplicate_keys.sum())

print("Number of repeated route-time keys:",
      df.loc[duplicate_keys, key_columns]
      .drop_duplicates()
      .shape[0])

if duplicate_keys.any():
    print("\nExamples of repeated keys:")
    print(
        df.loc[duplicate_keys, key_columns + ["skaits"]]
        .head(15)
        .to_string(index=False)
    )

# ==========================================
# 4. MONTHLY DATA COVERAGE
# ==========================================

print("\n" + "=" * 60)
print("MONTHLY DATA COVERAGE")
print("=" * 60)

df["month"] = df["datums"].dt.to_period("M").astype(str)

monthly_summary = (
    df.groupby("month")
    .agg(
        records=("skaits", "size"),
        total_qr_validations=("skaits", "sum"),
        active_routes=("route_clean", "nunique"),
        active_days=("datums", "nunique")
    )
)

print(monthly_summary.to_string())

monthly_summary.to_csv(
    REPORT_DIR / "monthly_data_summary.csv",
    encoding="utf-8-sig"
)

# ==========================================
# 5. DAILY COVERAGE
# ==========================================

print("\n" + "=" * 60)
print("DAILY COVERAGE")
print("=" * 60)

daily = (
    df.groupby("datums")
    .agg(
        records=("skaits", "size"),
        total_validations=("skaits", "sum")
    )
)

print("\nNumber of dates:", len(daily))
print("Dates with zero records:", (daily["records"] == 0).sum())
print("\nDaily record-count statistics:")
print(daily["records"].describe())

# ==========================================
# 6. FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("DEEP AUDIT COMPLETED")
print("=" * 60)

print("\nReports saved in:", REPORT_DIR)
print("- route_summary.csv")
print("- monthly_data_summary.csv")
