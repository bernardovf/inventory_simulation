import numpy as np
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

def load_forecast_vintages(path, site, product, site_col="Site", product_col="Product", date_col="Date", as_of_col="as_of_dt", forecast_col="Forecast"):
    """Load a forecast-vintages CSV (columns: Site, Product, Date, as_of_dt, Forecast).

    Each row is the forecast for `Date`, as it was generated on `as_of_dt` -
    the same target date typically appears many times, once per vintage that
    forecast it. Filters to the given site/product and returns a dict mapping
    each as_of_dt (Timestamp) to a pandas Series of forecasted quantity
    indexed by target Date, so a simulation standing "as of" a given day can
    look up the forecast that would actually have been available on that day,
    instead of one that peeks into the future.
    """
    forecast_df = pd.read_csv(path)
    forecast_df[site_col] = forecast_df[site_col].astype(str)
    forecast_df[product_col] = forecast_df[product_col].astype(str)
    forecast_df = forecast_df[forecast_df[site_col] == site]
    forecast_df = forecast_df[forecast_df[product_col] == product]

    if forecast_df.empty:
        raise ValueError(f"No forecast rows found for {site_col}={site!r}, {product_col}={product!r}")

    forecast_df[date_col] = pd.to_datetime(forecast_df[date_col])
    forecast_df[as_of_col] = pd.to_datetime(forecast_df[as_of_col])

    vintages = {}
    for as_of_dt, group in forecast_df.groupby(as_of_col):
        vintages[as_of_dt] = group.set_index(date_col)[forecast_col].astype(float).sort_index()

    return vintages

def forecast_windows_from_vintages(vintages, dates, lead_time, review_period, forecast_lag=0):
    """For each date, sum the forecast that would have been available "as of"
    that day over the upcoming lead-time / lead-time-plus-review-period
    windows - the real-forecast counterpart to the coefficient-of-variation
    synthetic forecast.

    For each date, uses the most recent vintage (as_of_dt) on or before
    (date - forecast_lag) - i.e. the freshest forecast actually usable that
    day once `forecast_lag` days of delay (production freeze, review cycle,
    pipeline latency, ...) between generating a forecast and being able to
    act on it are accounted for. forecast_lag=0 (the default) uses the
    freshest vintage available with no delay. Dates earlier than the first
    usable vintage fall back to that earliest vintage rather than peeking
    into a forecast from the future. If a vintage doesn't extend far enough
    forward to cover the full window, the missing days are filled with that
    vintage's own average forecasted value.

    Returns two numpy arrays aligned to `dates`: (forecast_lead_time_sums,
    forecast_protection_period_sums).
    """
    as_of_dates = np.array(sorted(vintages.keys()))
    dates = pd.to_datetime(pd.Index(dates))
    lag = pd.Timedelta(days=forecast_lag)

    lead_time_sums = np.zeros(len(dates))
    protection_period_sums = np.zeros(len(dates))

    for i, current_date in enumerate(dates):
        idx = np.searchsorted(as_of_dates, current_date - lag, side="right") - 1
        idx = max(idx, 0)
        vintage = vintages[as_of_dates[idx]]

        horizon_end = current_date + pd.Timedelta(days=lead_time + review_period)
        window = vintage[(vintage.index > current_date) & (vintage.index <= horizon_end)]
        avg_daily = window.mean() if len(window) > 0 else 0.0

        lead_time_end = current_date + pd.Timedelta(days=lead_time)
        lead_time_window = window[window.index <= lead_time_end]
        lead_time_sums[i] = lead_time_window.sum() + max(lead_time - len(lead_time_window), 0) * avg_daily
        protection_period_sums[i] = window.sum() + max((lead_time + review_period) - len(window), 0) * avg_daily

    return lead_time_sums, protection_period_sums

def latest_forecast_by_date(vintages, forecast_lag=0):
    """For each target date, the most recent forecast that was actually
    usable by then - i.e. the last prediction available before that date's
    actual demand happened, once `forecast_lag` days of delay between
    generating a forecast and being able to act on it are accounted for.

    Only vintages with as_of_dt <= Date - forecast_lag are considered (a
    forecast that isn't usable until after the fact doesn't count as a
    prediction). forecast_lag=0 (the default) uses the freshest forecast
    available with no delay. Returns a pandas Series of forecasted quantity
    indexed by Date.
    """
    lag = pd.Timedelta(days=forecast_lag)
    frames = [
        pd.DataFrame({"Date": series.index, "as_of_dt": as_of_dt, "Forecast": series.to_numpy()})
        for as_of_dt, series in vintages.items()
    ]
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df[long_df["as_of_dt"] <= long_df["Date"] - lag]

    latest_idx = long_df.groupby("Date")["as_of_dt"].idxmax()
    return long_df.loc[latest_idx].set_index("Date")["Forecast"].sort_index()

def load_site_product_parameters(path):
    """Load per-Site/Product simulation parameters.

    Expects one row per Site/Product with columns Site, Product, and
    (case/spacing-insensitive) Forecast Error Cov, Average Lead Time,
    Std Dev Lead Time, MOQ, SS Settings.
    """
    canonical_names = {
        "site": "Site",
        "product": "Product",
        "forecasterrorcov": "Forecast_Error_Cov",
        "averageleadtime": "Average_Lead_Time",
        "stddevleadtime": "Lead_Time_Std_Dev",
        "moq": "MOQ",
        "sssettings": "SS_Settings",
    }

    def normalize(col):
        return col.strip().lower().replace(" ", "").replace("_", "")

    params_df = pd.read_csv(path)
    rename_map = {
        col: canonical_names[normalize(col)]
        for col in params_df.columns
        if normalize(col) in canonical_names
    }
    params_df = params_df.rename(columns=rename_map)

    required_columns = list(canonical_names.values())
    missing_columns = [col for col in required_columns if col not in params_df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns in site/product parameters file: {missing_columns}")

    params_df["Site"] = params_df["Site"].astype(str)
    params_df["Product"] = params_df["Product"].astype(str)

    return params_df[required_columns]

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
