import pandas as pd
import matplotlib.pyplot as plt
from utils import bad_update_dates
import numpy as np

hist_inventory = pd.read_csv("historical_inventory.csv")
hist_inventory = hist_inventory[hist_inventory["plant_code"] == "USW1"]

def _cv(x):
    x = x.dropna()
    if len(x) < 2 or x.mean() == 0:
        return np.nan
    return x.std(ddof=1) / x.mean()

def _prepare_inventory_date(hist_inventory):
    hist_inventory = hist_inventory[["plant_code", "material_number", "calendar_date", "total_unrestricted_stock"]]

    # Make sure calendar_date is datetime
    hist_inventory["calendar_date"] = pd.to_datetime(hist_inventory["calendar_date"])

    # All unique plant/material combinations
    combinations = (hist_inventory[["plant_code", "material_number"]].drop_duplicates())

    # All dates
    dates = pd.DataFrame({"calendar_date": pd.date_range("2024-12-29", "2026-06-01", freq="D")})

    # Cartesian product
    combinations["key"] = 1
    dates["key"] = 1

    full_index = (combinations.merge(dates, on="key").drop(columns="key"))

    # Merge with original data
    hist_inventory = full_index.merge(
        hist_inventory,
        on=["plant_code", "material_number", "calendar_date"],
        how="left")

    # Fill missing values
    hist_inventory["plant_material_number"] = (
        hist_inventory["material_number"].astype(str)
        + "~"
        + hist_inventory["plant_code"])

    stock_columns = ["total_unrestricted_stock"]

    for col in stock_columns:
        if col in hist_inventory.columns:
            hist_inventory[col] = hist_inventory[col].fillna(0)

    # Optional: sort
    hist_inventory = hist_inventory.sort_values(["plant_code", "material_number", "calendar_date"]).reset_index(drop=True)

    bad_mask = hist_inventory["calendar_date"].isin(bad_update_dates)

    # Temporarily mark only the known bad dates as missing
    hist_inventory.loc[bad_mask, stock_columns] = pd.NA

    # Fill them using the last valid prior date within each plant/material
    hist_inventory[stock_columns] = (hist_inventory.groupby(["plant_code", "material_number"])[stock_columns].ffill())

    # Keep 0 only where there was no earlier valid inventory
    hist_inventory[stock_columns] = hist_inventory[stock_columns].fillna(0)

    return hist_inventory

def _get_replenishment_points(hist_inventory):
    hist_inventory = hist_inventory.sort_values(["plant_code", "material_number", "calendar_date"]).reset_index(drop=True)

    hist_inventory["previous_unrestricted_stock"] = (
        hist_inventory
        .groupby(["plant_code", "material_number"])["total_unrestricted_stock"]
        .shift(1))

    hist_inventory["inventory_change"] = (
            hist_inventory["total_unrestricted_stock"]
            - hist_inventory["previous_unrestricted_stock"])

    hist_inventory["is_replenishment"] = (
            hist_inventory["inventory_change"] > 0)

    hist_inventory["inferred_replenishment_qty"] = (
        hist_inventory["inventory_change"].clip(lower=0))

    return hist_inventory

def _get_replenishment_points_percentile(hist_inventory):
    group_cols = ["plant_code", "material_number"]

    hist_inventory = (
        hist_inventory
        .sort_values(group_cols + ["calendar_date"])
        .reset_index(drop=True)
        .copy())

    hist_inventory["previous_unrestricted_stock"] = (
        hist_inventory
        .groupby(group_cols)["total_unrestricted_stock"]
        .shift(1))

    hist_inventory["inventory_change"] = (
        hist_inventory["total_unrestricted_stock"]
        - hist_inventory["previous_unrestricted_stock"])

    positive_changes = hist_inventory.loc[hist_inventory["inventory_change"] > 0]

    thresholds = (
        positive_changes
        .groupby(group_cols)["inventory_change"]
        .agg(lower_threshold=lambda x: x.quantile(0.1),)
        .reset_index())

    hist_inventory = hist_inventory.merge(
        thresholds,
        on=group_cols,
        how="left")

    hist_inventory["is_replenishment"] = (
        hist_inventory["inventory_change"].gt(0)
        & hist_inventory["inventory_change"].ge(hist_inventory["lower_threshold"]))

    hist_inventory["inferred_replenishment_qty"] = (
        hist_inventory["inventory_change"]
        .where(hist_inventory["is_replenishment"], 0)
        .clip(lower=0))

    hist_inventory = hist_inventory.drop(columns=["lower_threshold"])

    return hist_inventory

def _get_replenishment_points_minima(hist_inventory):
    lookback_days = 7
    tolerance = 0.02
    group_cols = ["plant_code", "material_number"]

    hist_inventory = (
        hist_inventory
        .sort_values(group_cols + ["calendar_date"])
        .reset_index(drop=True)
        .copy()
    )

    hist_inventory["previous_unrestricted_stock"] = (
        hist_inventory
        .groupby(group_cols)["total_unrestricted_stock"]
        .shift(1)
    )

    hist_inventory["inventory_change"] = (
        hist_inventory["total_unrestricted_stock"]
        - hist_inventory["previous_unrestricted_stock"]
    )

    # Minimum stock observed before today within the recent lookback window
    hist_inventory["recent_min_stock"] = (
        hist_inventory
        .groupby(group_cols)["total_unrestricted_stock"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=lookback_days,
                min_periods=1
            ).min()
        )
    )

    # Yesterday is considered "near the local minimum"
    # when it is no more than tolerance above that minimum.
    hist_inventory["near_local_minimum"] = (
        hist_inventory["previous_unrestricted_stock"]
        <= hist_inventory["recent_min_stock"] * (1 + tolerance)
    )

    hist_inventory["is_replenishment"] = (
        (hist_inventory["inventory_change"] > 0)
        & hist_inventory["near_local_minimum"]
    )

    hist_inventory["inferred_replenishment_qty"] = (
        hist_inventory["inventory_change"]
        .where(hist_inventory["is_replenishment"], 0)
        .clip(lower=0)
    )

    return hist_inventory.drop(
        columns=["recent_min_stock", "near_local_minimum"]
    )

def _create_histogram_data(hist_inventory):
    records = []

    for (plant, material), group in hist_inventory.groupby(
        ["plant_code", "material_number"]
    ):
        x = group.loc[group["inventory_change"] > 0, "inventory_change"].dropna().values

        if len(x) < 2:
            continue

        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25

        if iqr == 0:
            bins = max(5, int(np.sqrt(len(x))))
        else:
            bin_width = 2 * iqr / (len(x) ** (1 / 3))
            bins = max(5, int(np.ceil((x.max() - x.min()) / bin_width)))

        counts, edges = np.histogram(x, bins=bins)

        for i in range(len(counts)):
            records.append({
                "plant_code": plant,
                "material_number": material,
                "bin_start": edges[i],
                "bin_end": edges[i + 1],
                "bin_center": (edges[i] + edges[i + 1]) / 2,
                "count": counts[i]
            })

    return pd.DataFrame(records)

def _safe_cv(series):
    series = pd.to_numeric(series, errors="coerce").dropna()

    if len(series) < 2:
        return np.nan

    mean = series.mean()

    if mean == 0:
        return np.nan

    return series.std(ddof=1) / abs(mean)

def _infer_inventory_policy(hist_inventory, min_events=4):
    group_cols = ["plant_code", "material_number"]

    df = (hist_inventory
        .sort_values(group_cols + ["calendar_date"])
        .copy())

    df["calendar_date"] = pd.to_datetime(df["calendar_date"])

    # Inventory immediately before and after each inferred receipt
    df["stock_before"] = df.groupby(group_cols)["total_unrestricted_stock"].shift(1)

    df["stock_after"] = df["total_unrestricted_stock"]

    events = df.loc[df["is_replenishment"]].copy()

    events["days_since_previous_replenishment"] = (
        events
        .groupby(group_cols)["calendar_date"]
        .diff()
        .dt.days
    )

    summaries = []

    for keys, group in events.groupby(group_cols):
        plant_code, material_number = keys

        n_events = len(group)

        if n_events < min_events:
            summaries.append({
                "plant_code": plant_code,
                "material_number": material_number,
                "number_of_replenishments": n_events,
                "inferred_policy": "Insufficient history"
            })
            continue

        qty_cv = _safe_cv(group["inferred_replenishment_qty"])
        before_cv = _safe_cv(group["stock_before"])
        after_cv = _safe_cv(group["stock_after"])
        interval_cv = _safe_cv(group["days_since_previous_replenishment"])

        median_q = group["inferred_replenishment_qty"].median()
        median_s = group["stock_before"].median()
        median_S = group["stock_after"].median()
        median_R = group["days_since_previous_replenishment"].median()

        # These cutoffs should be calibrated to your data
        quantity_stable = pd.notna(qty_cv) and qty_cv <= 0.25
        before_stable = pd.notna(before_cv) and before_cv <= 0.25
        after_stable = pd.notna(after_cv) and after_cv <= 0.25
        interval_regular = pd.notna(interval_cv) and interval_cv <= 0.20

        if quantity_stable and before_stable and not interval_regular:
            policy = "(s, Q)"

        elif interval_regular and after_stable and not quantity_stable:
            policy = "(R, S)"

        elif before_stable and after_stable and not interval_regular:
            policy = "(s, S)"

        elif interval_regular and quantity_stable:
            policy = "(R, s, Q)"

        elif interval_regular and after_stable:
            policy = "(R, s, S)"

        else:
            policy = "Unclear / irregular"

        summaries.append({
            "plant_code": plant_code,
            "material_number": material_number,
            "number_of_replenishments": n_events,
            "inferred_policy": policy,

            "estimated_s": median_s,
            "estimated_Q": median_q,
            "estimated_S": median_S,
            "estimated_R_days": median_R,

            "replenishment_qty_cv": qty_cv,
            "pre_replenishment_stock_cv": before_cv,
            "post_replenishment_stock_cv": after_cv,
            "replenishment_interval_cv": interval_cv
        })

    return pd.DataFrame(summaries)

hist_inventory = _prepare_inventory_date(hist_inventory)
hist_inventory = _get_replenishment_points_percentile(hist_inventory)
histogram_data = _create_histogram_data(hist_inventory)
metrics = _infer_inventory_policy(hist_inventory)

hist_inventory.to_csv("hist_inventory.csv", index=False)
histogram_data.to_csv("histogram_data.csv", index=False)
metrics.to_csv("metrics.csv", index=False)