import numpy as np
import pandas as pd

def load_historical_demand(path, site=None, site_col="Site", period_col="Period", date_col="Date", quantity_col="Quantity"):
    """Load a historical demand CSV (columns: Site, Period, Date, Quantity).

    Returns a 1-D array of demand quantities ordered by Period, with any gaps
    in the period sequence filled in as zero demand. Raises if the file
    contains more than one site and `site` isn't given to pick one.
    """
    demand_df = pd.read_csv(path)
    demand_df[date_col] = pd.to_datetime(demand_df[date_col])

    if site is not None:
        demand_df = demand_df[demand_df[site_col] == site]

    sites = demand_df[site_col].unique()
    if len(sites) != 1:
        raise ValueError(
            f"Expected exactly one site in the historical demand data, got {sorted(sites)}. "
            "Pass `site=` to select one.")

    demand_df = demand_df.sort_values(period_col)

    full_periods = pd.DataFrame({
        period_col: np.arange(demand_df[period_col].min(), demand_df[period_col].max() + 1)
    })
    demand_df = full_periods.merge(demand_df, on=period_col, how="left")
    demand_df[quantity_col] = demand_df[quantity_col].fillna(0)

    return demand_df[quantity_col].to_numpy(dtype=float)

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
