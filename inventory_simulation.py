import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils import load_historical_demand

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

# --- Demand source -----------------------------------------------------
# Leave `historical_demand_csv` as None to simulate with synthetic demand
# (`demand_average` / `demand_std_deviation` below). To use real demand
# instead, set it to a CSV path with columns Site, Product, Date, Demand,
# along with the `historical_demand_site` / `historical_demand_product` to
# filter to - in that case only `forecast_error_cov` is needed, since the
# forecast is generated from the historical demand rather than from
# `demand_average`.
historical_demand_csv = "historical_demand.csv"  # e.g. "historical_demand.csv"
historical_demand_site = "US10"  # e.g. "US10"
historical_demand_product = "1001125"  # e.g. "1001125"

if historical_demand_csv:
    demand_history = load_historical_demand(historical_demand_csv, site=historical_demand_site, product=historical_demand_product)
    time = len(demand_history)
    demand_average = demand_history.mean()
    historical_dates = demand_history.index.to_numpy()
else:
    time = 500
    demand_average = 100
    demand_std_deviation = 75
    historical_dates = None

warm_up_period = 90
forecast_error_cov = 0.32  # std dev of forecast error, as a fraction of average demand
init_on_hand = 30000
average_lead_time = 39
lead_time_std_dev = 1
MOQ = 18188
review_period = 7
plot_historical_inventory = False

safety_stock_range = range(10000, 80000, 10000)

n_simulations = 100  # Monte Carlo replications to average per (safety_stock, policy)

results = {}
fill_rate_by_ss = {}
for i, safety_stock_units in enumerate(safety_stock_range):
    fill_rate_by_ss[safety_stock_units] = 0

for sim in range(n_simulations):
    if historical_demand_csv:
        demand = demand_history
    else:
        demand = np.maximum(rng.normal(demand_average, demand_std_deviation, time).round(), 0)

    # The forecast is a noisy estimate of demand - it drives the reorder point / order-up-to
    # level, while the simulation itself is still driven by actual demand above.
    forecast_error_std = forecast_error_cov * demand_average
    forecast = np.maximum(demand + rng.normal(0, forecast_error_std, time).round(), 0)

    for i, safety_stock_units in enumerate(safety_stock_range):
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

            for name, df in results.items():
                x = historical_dates[df["Period"].to_numpy()] if historical_dates is not None else df["Period"]
                ax.plot(x, df["On_Hand"], label=name, linewidth=2)

            ax.set_ylim(bottom=0)
            ax.set_xlabel("Date" if historical_dates is not None else "Period")
            ax.set_ylabel("On Hand")
            ax.grid(alpha=0.5)
            ax.legend()

            if historical_dates is not None:
                fig.autofmt_xdate()

            plt.tight_layout()
            plt.show()



for i, safety_stock_units in enumerate(safety_stock_range):
    print(f"{safety_stock_units}: {fill_rate_by_ss[safety_stock_units]:.1%}")
print("")

x_values = list(safety_stock_range)
y_values = [fill_rate_by_ss[ss] for ss in x_values]

fig_fill_rate, ax_fill_rate = plt.subplots(figsize=(10, 6))

ax_fill_rate.plot(
    x_values,
    y_values,
    marker="o",
    linewidth=2
)

ax_fill_rate.set_xlabel("Safety stock (units)")
ax_fill_rate.set_ylabel("Fill rate")
ax_fill_rate.set_title("Fill rate vs. Safety Stock")
ax_fill_rate.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax_fill_rate.set_ylim(0, 1.01)
ax_fill_rate.grid(alpha=0.5)

fig_fill_rate.tight_layout()
fig_fill_rate.savefig(
    "fill_rate_by_safety_stock.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()