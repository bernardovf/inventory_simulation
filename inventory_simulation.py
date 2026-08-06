import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plot_historical_inventory = False

from utils import load_historical_demand, load_site_product_parameters, load_forecast_vintages, forecast_windows_from_vintages

def simulate_inventory(demand, forecast, lead_time, initial_on_hand, safety_stock, MOQ=None, Review_Period=None, lead_time_std_dev=0, rng=None, forecast_lead_time_series=None, forecast_protection_period_series=None):
    if rng is None:
        rng = np.random.default_rng()

    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    periods = len(demand)

    # A real forecast (one lead-time/protection-period sum per day, looked up
    # from what was actually known "as of" that day) takes priority over the
    # coefficient-of-variation synthetic forecast below when supplied.
    using_real_forecast = forecast_lead_time_series is not None
    if using_real_forecast:
        forecast_lead_time_series = np.asarray(forecast_lead_time_series, dtype=float)
        forecast_protection_period_series = np.asarray(forecast_protection_period_series, dtype=float)

    # Extra space is needed for orders arriving after the simulation horizon;
    # padded generously so a long random lead time doesn't fall off the end.
    max_lead_time = lead_time + round(4 * lead_time_std_dev)
    scheduled_receipts = np.zeros(periods + max_lead_time + 1, dtype=float)

    on_hand = np.zeros(periods)
    receipts = np.zeros(periods)
    order_qty = np.zeros(periods)
    fulfilled_demand = np.zeros(periods)
    lost_sales = np.zeros(periods)

    inventory_position_before_order = np.zeros(periods)
    inventory_position_after_order = np.zeros(periods)
    reorder_point = np.zeros(periods)
    order_up_to_level = np.zeros(periods)
    realized_lead_time = np.zeros(periods)

    for t in range(periods):
        # 1. Receive orders due today
        receipts[t] = scheduled_receipts[t]

        if t == 0:
            available = initial_on_hand + receipts[t]
        else:
            available = on_hand[t - 1] + receipts[t]

        # 2. Satisfy demand
        fulfilled_demand[t] = min(available, demand[t])
        lost_sales[t] = demand[t] - fulfilled_demand[t]

        on_hand[t] = available - fulfilled_demand[t]

        # Orders already scheduled after today
        pipeline = scheduled_receipts[t + 1:].sum()

        # 3. Inventory position before today's new order
        inventory_position_before_order[t] = on_hand[t] + pipeline

        ip = inventory_position_before_order[t]

        # 4. Apply the selected policy
        qty = 0

        # Demand over the upcoming lead time / protection period, taken from the
        # forecast (not the actual future demand, which wouldn't be known yet).
        if using_real_forecast:
            forecast_lead_time = forecast_lead_time_series[t]
            forecast_protection_period = forecast_protection_period_series[t]
        else:
            future_forecast = forecast[t + 1:]
            forecast_lead_time = future_forecast[:lead_time].sum()
            forecast_protection_period = future_forecast[:lead_time + Review_Period].sum()

        s = forecast_lead_time + safety_stock
        S = forecast_protection_period + safety_stock

        reorder_point[t] = s
        order_up_to_level[t] = S

        if Review_Period == 0:
            if ip <= s:
                qty = max(MOQ, S - ip)
        else:
            if t % Review_Period == 0 and ip <= s:
                qty = max(MOQ, S - ip)

        order_qty[t] = qty

        # 5. Schedule the order receipt
        if qty > 0:
            if lead_time_std_dev > 0:
                actual_lead_time = max(1, round(rng.normal(lead_time, lead_time_std_dev)))
            else:
                actual_lead_time = lead_time

            realized_lead_time[t] = actual_lead_time
            arrival_period = t + actual_lead_time

            if arrival_period < len(scheduled_receipts):
                scheduled_receipts[arrival_period] += qty

        inventory_position_after_order[t] = ip + qty

    results = pd.DataFrame({
        "Period": np.arange(periods),
        "Demand": demand,
        "Forecast": forecast,
        "Receipts": receipts,
        "Fulfilled_Demand": fulfilled_demand,
        "Lost_Sales": lost_sales,
        "On_Hand": on_hand,
        "Order_Qty": order_qty,
        "Lead_Time": realized_lead_time,
        "Reorder_Point": reorder_point,
        "Order_Up_To_Level": order_up_to_level,
        "Inventory_Position_Before_Order": inventory_position_before_order,
        "Inventory_Position_After_Order": inventory_position_after_order
    })

    return results

def simulate_all_items(warm_up_period, review_period, safety_stock_steps, n_simulations, parameters, forecast_vintages_csv=None):
    output_rows = []

    for _, param_row in parameters.iterrows():
        site = param_row["Site"]
        product = param_row["Product"]
        print(site, product)
        forecast_error_cov = param_row["Forecast_Error_Cov"]
        average_lead_time = int(round(param_row["Average_Lead_Time"]))
        lead_time_std_dev = param_row["Lead_Time_Std_Dev"]
        MOQ = param_row["MOQ"]
        ss_settings = param_row["SS_Settings"]
        init_on_hand = param_row["SS_Settings"]

        safety_stock_range = np.linspace(ss_settings / 10, ss_settings * 3, safety_stock_steps)

        demand_history = load_historical_demand(historical_demand_csv, site=site, product=product)
        time = len(demand_history)
        demand_average = demand_history.mean()
        demand = demand_history
        historical_dates = demand_history.index.to_numpy()

        # Real forecast: use the forecast vintage that would actually have been
        # available on each historical day, instead of implying it from demand
        # plus a coefficient-of-variation noise term. Computed once per item
        # since it doesn't depend on the Monte Carlo replication or safety stock.
        if forecast_vintages_csv is not None:
            vintages = load_forecast_vintages(forecast_vintages_csv, site=site, product=product)
            forecast_lead_time_series, forecast_protection_period_series = forecast_windows_from_vintages(
                vintages, demand_history.index, average_lead_time, review_period)
        else:
            forecast_lead_time_series = None
            forecast_protection_period_series = None

        fill_rate_by_ss = {ss: 0.0 for ss in safety_stock_range}

        for sim in range(n_simulations):
            if forecast_vintages_csv is None:
                forecast_error_std = forecast_error_cov * demand_average
                forecast = np.maximum(demand + rng.normal(0, forecast_error_std, time).round(), 0)
            else:
                forecast = np.full(time, np.nan)  # not used for s/S here; real forecast doesn't reduce to one series

            for safety_stock_units in safety_stock_range:
                results = simulate_inventory(
                    demand=demand,
                    forecast=forecast,
                    lead_time=average_lead_time,
                    lead_time_std_dev=lead_time_std_dev,
                    initial_on_hand=init_on_hand,
                    safety_stock=safety_stock_units,
                    MOQ=MOQ,
                    Review_Period=review_period,
                    rng=rng,
                    forecast_lead_time_series=forecast_lead_time_series,
                    forecast_protection_period_series=forecast_protection_period_series)

                results = results[results["Period"] > warm_up_period]
                total_demand = results["Demand"].sum()
                total_fulfilled = results["Fulfilled_Demand"].sum()
                fill_rate = (total_fulfilled / total_demand
                             if total_demand > 0
                             else float("nan"))
                fill_rate_by_ss[safety_stock_units] += fill_rate / n_simulations

                if plot_historical_inventory:
                    fig, ax = plt.subplots(figsize=(12, 6))

                    x = historical_dates[results["Period"].to_numpy()]
                    ax.plot(x, results["On_Hand"], linewidth=2)

                    ax.set_ylim(bottom=0)
                    ax.set_xlabel("Date")
                    ax.set_ylabel("On Hand")
                    ax.set_title(f"{site} / {product} - Safety stock {safety_stock_units}")
                    ax.grid(alpha=0.5)
                    fig.autofmt_xdate()

                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)

        for safety_stock_units in safety_stock_range:
            output_rows.append({
                "Site": site,
                "Product": product,
                "Safety Stock Units": round(safety_stock_units, 0),
                "Fill Rate": fill_rate_by_ss[safety_stock_units]
            })

    return pd.DataFrame(output_rows)

def simulate_combo(site_chosen, product_chosen, warm_up_period, review_period, safety_stock_units, parameters, forecast_vintages_csv=None):
    for _, param_row in parameters.iterrows():
        if param_row["Site"] == site_chosen and param_row["Product"] == product_chosen:
            site = param_row["Site"]
            product = param_row["Product"]
            print(site, product)
            forecast_error_cov = param_row["Forecast_Error_Cov"]
            average_lead_time = int(round(param_row["Average_Lead_Time"]))
            lead_time_std_dev = param_row["Lead_Time_Std_Dev"]
            MOQ = param_row["MOQ"]
            ss_settings = param_row["SS_Settings"]
            init_on_hand = param_row["SS_Settings"]

            demand_history = load_historical_demand(historical_demand_csv, site=site, product=product)
            time = len(demand_history)
            demand_average = demand_history.mean()
            demand = demand_history
            historical_dates = demand_history.index.to_numpy()

            if forecast_vintages_csv is None:
                forecast_error_std = forecast_error_cov * demand_average
                forecast = np.maximum(demand + rng.normal(0, forecast_error_std, time).round(), 0)
                forecast_lead_time_series = None
                forecast_protection_period_series = None
            else:
                vintages = load_forecast_vintages(forecast_vintages_csv, site=site, product=product)
                forecast_lead_time_series, forecast_protection_period_series = forecast_windows_from_vintages(
                    vintages, demand_history.index, average_lead_time, review_period)
                forecast = np.full(time, np.nan)  # not used for s/S here; real forecast doesn't reduce to one series

            results = simulate_inventory(
                demand=demand,
                forecast=forecast,
                lead_time=average_lead_time,
                lead_time_std_dev=lead_time_std_dev,
                initial_on_hand=init_on_hand,
                safety_stock=safety_stock_units,
                MOQ=MOQ,
                Review_Period=review_period,
                rng=rng,
                forecast_lead_time_series=forecast_lead_time_series,
                forecast_protection_period_series=forecast_protection_period_series)

            results = results[results["Period"] > warm_up_period]
            total_demand = results["Demand"].sum()
            total_fulfilled = results["Fulfilled_Demand"].sum()
            fill_rate = (total_fulfilled / total_demand
                         if total_demand > 0
                         else float("nan"))
            print(fill_rate)

            fig, ax = plt.subplots(figsize=(12, 6))

            x = historical_dates[results["Period"].to_numpy()]
            ax.plot(x, results["On_Hand"], linewidth=2)

            ax.set_ylim(bottom=0)
            ax.set_xlabel("Date")
            ax.set_ylabel("On Hand")
            ax.set_title(f"{site} / {product} - Safety stock {safety_stock_units}")
            ax.grid(alpha=0.5)
            fig.autofmt_xdate()

            plt.tight_layout()
            plt.show()
            plt.close(fig)


rng = np.random.default_rng(42)

historical_demand_csv = "historical_demand.csv"
site_product_parameters_csv = "site_product_parameters.csv"
forecast_vintages_csv = "forecast_vintages.csv"  # columns: Site, Product, Date, as_of_dt, Forecast; set to None to use the Forecast_Error_Cov synthetic forecast instead
output_csv = "fill_rate_by_site_product.csv"

warm_up_period = 30
review_period = 7
safety_stock_steps = 20  # number of safety stock levels to simulate, from SS Settings / 2 to SS Settings * 2
n_simulations = 100  # Monte Carlo replications to average per (safety_stock, policy)
parameters = load_site_product_parameters(site_product_parameters_csv)

#output_df = simulate_all_items(warm_up_period, review_period, safety_stock_steps, n_simulations, parameters, forecast_vintages_csv=forecast_vintages_csv)
output_df = simulate_combo("USW1", "5071379", warm_up_period, review_period, 6894, parameters, forecast_vintages_csv=forecast_vintages_csv)

#output_df.to_csv(output_csv, index=False)