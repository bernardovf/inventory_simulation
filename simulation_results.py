import numpy as np
import pandas as pd

# Read files
curve = pd.read_csv("fill_rate_by_site_product.csv")
master = pd.read_csv("Site Product Master.csv")

# Standardize keys
curve["Site"] = curve["Site"].astype(str).str.strip()
master["Site"] = master["Site"].astype(str).str.strip()

curve["Product"] = curve["Product"].astype(str).str.strip()
master["Product"] = master["Product"].astype(str).str.strip()

# Helper to convert numeric columns
def to_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )

# Convert numeric columns
master["Units"] = to_numeric(master["Units"])
master["SS Settings"] = to_numeric(master["SS Settings"])
master["Target Fill Rate"] = to_numeric(master["Target Fill Rate"])

curve["Safety Stock Units"] = to_numeric(curve["Safety Stock Units"])
curve["Fill Rate"] = to_numeric(curve["Fill Rate"])

curve = (
    curve.dropna(subset=["Site", "Product", "Safety Stock Units", "Fill Rate"])
         .sort_values(["Site", "Product", "Safety Stock Units"])
)

# Build lookup
curve_lookup = {
    (site, product): grp.groupby("Safety Stock Units", as_index=False)["Fill Rate"]
                        .mean()
                        .sort_values("Safety Stock Units")
    for (site, product), grp in curve.groupby(["Site", "Product"])
}

current_fill_rates = []
model_fill_rates = []

# Keep only Site/Product combinations that exist in the curve
master = master[
    master.apply(
        lambda row: (row["Site"], row["Product"]) in curve_lookup,
        axis=1
    )
].copy()

current_fill_rates = []
model_fill_rates = []

for _, row in master.iterrows():

    key = (row["Site"], row["Product"])
    grp = curve_lookup[key]

    current_fill_rates.append(
        np.interp(
            row["SS Settings"],
            grp["Safety Stock Units"],
            grp["Fill Rate"]
        )
    )

    model_fill_rates.append(
        np.interp(
            row["Units"],
            grp["Safety Stock Units"],
            grp["Fill Rate"]
        )
    )

master["Current Fill Rate"] = current_fill_rates
master["Model Fill Rate"] = model_fill_rates

output = master[
    [
        "Site",
        "Product",
        "Units",
        "SS Settings",
        "Target Fill Rate",
        "Current Fill Rate",
        "Model Fill Rate",
    ]
].copy()
output = output["Target Fill Rate"] / 100
output.to_csv("site_product_fill_rates.csv", index=False)

print(output.head())
print(f"Rows exported: {len(output):,}")