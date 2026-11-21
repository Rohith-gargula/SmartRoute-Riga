import pandas as pd
df = pd.read_csv(
    "data/processed/combined_qr_validations.csv",
    low_memory=False
)
route95 = df[
    (df["transports"] == "Autobuss") &
    (df["marsruts"].astype(str).str.strip() == "95")
]
report = pd.DataFrame([{
    "transport": "Autobuss",
    "route": "95",
    "source_records": len(route95),
    "total_qr_validations": int(route95["skaits"].sum()),
    "reason_for_exclusion": (
        "Only four isolated observations; "
        "insufficient consecutive hourly records "
        "for next-hour forecasting"
    )
}])
report.to_csv(
    "reports/tables/route_exclusion_report.csv",
    index=False
)
print(report.to_string(index=False))
print("\nRoute exclusion report saved.")
