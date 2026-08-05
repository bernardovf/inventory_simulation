import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils import load_historical_demand, load_site_product_parameters

def simulate_inventory(demand, forecast, lead_time, initial_on_hand, safety_stock, MOQ=None, Review_Period=None, lead_time_std_dev=0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    demand = np.asarray(demand, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    periods = len(demand)

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

rng = np.random.default_rng(42)

# --- Data sources --------------------------------------------------------
# Real demand history (columns: Site, Product, Date, Demand).
historical_demand_csv = "historical_demand.csv"
# Per-Site/Product parameters (columns: Site, Product, Forecast Error Cov,
# Average Lead Time, Std Dev Lead Time, MOQ, SS Settings) - one row per
# Site/Product to simulate. SS Settings is the site/product's current
# safety stock setting; it's swept from half to double that value.
site_product_parameters_csv = "site_product_parameters.csv"
output_csv = "fill_rate_by_site_product.csv"

warm_up_period = 90
init_on_hand = 30000
review_period = 7
plot_historical_inventory = False

safety_stock_steps = 20  # number of safety stock levels to simulate, from SS Settings / 2 to SS Settings * 2
n_simulations = 100  # Monte Carlo replications to average per (safety_stock, policy)

parameters = load_site_product_parameters(site_product_parameters_csv)

output_rows = []

for _, param_row in parameters.iterrows():
    site = param_row["Site"]
    product = param_row["Product"]
    forecast_error_cov = param_row["Forecast_Error_Cov"]
    average_lead_time = int(round(param_row["Average_Lead_Time"]))
    lead_time_std_dev = param_row["Lead_Time_Std_Dev"]
    MOQ = param_row["MOQ"]
    ss_settings = param_row["SS_Settings"]

    safety_stock_range = np.linspace(ss_settings / 2, ss_settings * 2, safety_stock_steps)

    demand_history = load_historical_demand(historical_demand_csv, site=site, product=product)
    time = len(demand_history)
    demand_average = demand_history.mean()
    demand = demand_history
    historical_dates = demand_history.index.to_numpy()

    fill_rate_by_ss = {ss: 0.0 for ss in safety_stock_range}

    for sim in range(n_simulations):
        # The forecast is a noisy estimate of demand - it drives the reorder point / order-up-to
        # level, while the simulation itself is still driven by actual demand above.
        forecast_error_std = forecast_error_cov * demand_average
        forecast = np.maximum(demand + rng.normal(0, forecast_error_std, time).round(), 0)

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
                rng=rng)

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

    print(f"{site} / {product}:")
    for safety_stock_units in safety_stock_range:
        print(f"  {safety_stock_units}: {fill_rate_by_ss[safety_stock_units]:.1%}")
        output_rows.append({
            "Site": site,
            "Product": product,
            "Safety Stock Units": safety_stock_units,
            "Fill Rate": fill_rate_by_ss[safety_stock_units]
        })
    print("")

    fig_fill_rate, ax_fill_rate = plt.subplots(figsize=(10, 6))

    ax_fill_rate.plot(
        list(safety_stock_range),
        [fill_rate_by_ss[ss] for ss in safety_stock_range],
        marker="o",
        linewidth=2
    )

    ax_fill_rate.set_xlabel("Safety stock (units)")
    ax_fill_rate.set_ylabel("Fill rate")
    ax_fill_rate.set_title(f"Fill rate vs. Safety Stock - {site} / {product}")
    ax_fill_rate.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax_fill_rate.set_ylim(0, 1.01)
    ax_fill_rate.grid(alpha=0.5)

    fig_fill_rate.tight_layout()
    fig_fill_rate.savefig(
        f"fill_rate_by_safety_stock_{site}_{product}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig_fill_rate)

pd.DataFrame(output_rows).to_csv(output_csv, index=False)