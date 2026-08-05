import pandas as pd

def load_historical_demand(path, site, product, site_col="Site", product_col="Product", date_col="Date", quantity_col="Demand"):
    """Load a historical demand CSV (columns: Site, Product, Date, Demand).

    Filters to the given site/product first, then returns a pandas Series of
    daily demand quantities indexed by Date, spanning every day from the
    earliest to the latest date found for that site/product, with any day
    missing from the file treated as zero demand.
    """
    demand_df = pd.read_csv(path)
    demand_df["Site"] = demand_df["Site"].astype(str)
    demand_df["Product"] = demand_df["Product"].astype(str)
    demand_df = demand_df[demand_df[site_col] == site]
    demand_df = demand_df[demand_df[product_col] == product]

    if demand_df.empty:
        raise ValueError(f"No rows found for {site_col}={site!r}, {product_col}={product!r}")

    demand_df[date_col] = pd.to_datetime(demand_df[date_col])
    demand_df = demand_df.sort_values(date_col)

    full_dates = pd.DataFrame({
        date_col: pd.date_range(demand_df[date_col].min(), demand_df[date_col].max(), freq="D")
    })
    demand_df = full_dates.merge(demand_df, on=date_col, how="left")
    demand_df[quantity_col] = demand_df[quantity_col].fillna(0)

    return demand_df.set_index(date_col)[quantity_col].astype(float)

bad_update_dates = pd.to_datetime([
    "2025-01-20",
    "2025-02-09",
    "2025-03-08",
    "2025-04-05",
    "2025-05-03",
    "2025-05-31",
    "2025-07-06",
    "2025-07-07",
    "2025-08-02",
    "2025-08-30",
    "2025-09-27",
    "2025-10-26",
    "2025-11-02",
    "2025-11-22",
    "2026-02-07",
    "2026-03-07",
    "2026-04-04",
    "2026-05-02",
    "2026-05-25",
    "2026-05-30",
    "2026-07-05",
])
