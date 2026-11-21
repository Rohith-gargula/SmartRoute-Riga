
from pathlib import Path
import pandas as pd

# ==================================================
# SMARTROUTE-RIGA: DATA INSPECTION
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports" / "tables"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("SMARTROUTE-RIGA: DATA INSPECTION")
print("=" * 65)

# 1. Find monthly CSV files
csv_files = sorted(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

print(f"\nMonthly files found: {len(csv_files)}")

# 2. Load and combine all files
dataframes = []

for file in csv_files:
    print(f"Reading: {file.name}")

    try:
        df = pd.read_csv(
            file,
            sep=";",
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            file,
            sep=";",
            encoding="cp1257"
        )

    df.columns = df.columns.str.strip()
    df["source_file"] = file.name
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

print("\nAll monthly files combined successfully.")

# 3. Dataset overview
print("\n" + "=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)

print("\nRows:", len(data))
print("Columns:", len(data.columns))
print("\nColumn names:")
print(data.columns.tolist())

print("\nFirst five records:")
print(data.head().to_string(index=False))

print("\nData types:")
print(data.dtypes)

# 4. Standardise column names
data.columns = [
    str(column).strip().lower()
    for column in data.columns
]

required_columns = [
    "datums",
    "stunda",
    "transports",
    "marsruts",
    "skaits"
]

missing_columns = [
    column for column in required_columns
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}. "
        f"Actual columns: {data.columns.tolist()}"
    )

# 5. Data quality
print("\n" + "=" * 65)
print("DATA QUALITY")
print("=" * 65)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate complete records:", data.duplicated().sum())

# 6. Convert data types
data["skaits"] = pd.to_numeric(
    data["skaits"],
    errors="coerce"
)

data["datums"] = pd.to_datetime(
    data["datums"],
    format="%Y.%m.%d",
    errors="coerce"
)

print("\nInvalid dates:", data["datums"].isna().sum())
print("Invalid validation counts:", data["skaits"].isna().sum())

# 7. Validation statistics
print("\n" + "=" * 65)
print("VALIDATION STATISTICS")
print("=" * 65)

print("\nQR validation count statistics:")
print(data["skaits"].describe())

print("\nTransport types:")
print(data["transports"].value_counts(dropna=False))

print("\nTop 15 routes by total QR validations:")
print(
    data.groupby("marsruts")["skaits"]
    .sum()
    .sort_values(ascending=False)
    .head(15)
)

# 8. Date coverage
print("\nDate coverage:")
print("First date:", data["datums"].min())
print("Last date:", data["datums"].max())

# 9. Save combined dataset
output_file = PROCESSED_DIR / "combined_qr_validations.csv"

data.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

# 10. Save quality report
quality_report = pd.DataFrame({
    "column": data.columns,
    "data_type": [str(dtype) for dtype in data.dtypes],
    "missing_values": data.isnull().sum().values,
    "missing_percentage": (
        data.isnull().mean().values * 100
    ).round(2)
})

quality_report.to_csv(
    REPORT_DIR / "data_quality_report.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 65)
print("INSPECTION COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nCombined dataset:")
print(output_file)

print("\nQuality report:")
print(REPORT_DIR / "data_quality_report.csv")
