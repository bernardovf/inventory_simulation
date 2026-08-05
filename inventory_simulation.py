import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_historical_demand

def simulate_inventory(demand, forecast, policy, lead_time, initial_on_hand, safety_stock, MOQ=None, Review_Period=None, lead_time_std_dev=0, rng=None):
    valid_policies = {
        "(s,Q)",
        "(R,S)",
        "(s,S)",
        "(R,s,S)",
        "(R,s,Q)"}

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

        policies = ["(C,MOQ)", "(C,No MOQ)", "(P,MOQ)", "(P,Non MOQ)"]

        if policy == "(C,MOQ)":
            if ip <= s:
                qty = max(MOQ, S - ip)

        elif policy == "(C,No MOQ)":
            if ip <= s:
                qty = max(0, S - ip)

        elif policy == "(P,MOQ)":
            if t % Review_Period == 0 and ip <= s:
                qty = max(MOQ, S - ip)

        elif policy == "(P,Non MOQ)":
            if t % Review_Period == 0 and ip <= s:
                qty = max(0, S - ip)

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
else:
    time = 500
    demand_average = 100
    demand_std_deviation = 75

warm_up_period = 60
forecast_error_cov = 0.32  # std dev of forecast error, as a fraction of average demand
init_on_hand = 30000
average_lead_time = 39
lead_time_std_dev = 1
MOQ = 18188
review_period = 7
plot_historical_inventory = True

policies = ["(P,MOQ)"]
safety_stock_range = range(10000, 60000, 10000)
safety_stock_range = range(32000, 33000, 10000)

n_simulations = 20  # Monte Carlo replications to average per (safety_stock, policy)

results = {}
fill_rate_sum = {pol: [0.0] * len(safety_stock_range) for pol in policies}

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
        for pol in policies:
            results_pol = simulate_inventory(
                demand=demand,
                forecast=forecast,
                policy=pol,
                lead_time=average_lead_time,
                lead_time_std_dev=lead_time_std_dev,
                initial_on_hand=init_on_hand,
                safety_stock=safety_stock_units,
                MOQ=MOQ,
                Review_Period=review_period,
                rng=rng)

            results_pol = results_pol[results_pol["Period"] > warm_up_period]
            total_demand = results_pol["Demand"].sum()
            total_fulfilled = results_pol["Fulfilled_Demand"].sum()
            fill_rate = (total_fulfilled / total_demand
                if total_demand > 0
                else float("nan"))
            fill_rate_sum[pol][i] += fill_rate
            results[pol] = results_pol

            if plot_historical_inventory:
                fig, ax = plt.subplots(figsize=(12, 6))

                for name, df in results.items():
                    ax.plot(
                        df["Period"],
                        df["On_Hand"],
                        label=name,
                        linewidth=2
                    )

                ax.set_ylim(bottom=0)
                ax.set_xlabel("Period")
                ax.set_ylabel("On Hand")
                ax.grid(alpha=0.5)
                ax.legend()

                plt.tight_layout()
                plt.show()
            exit()

fill_rate_by_policy = {
    pol: [total / n_simulations for total in fill_rate_sum[pol]]
    for pol in policies}

for i, safety_stock_units in enumerate(safety_stock_range):
    for pol in policies:
        print(f"{safety_stock_units}, {pol}: {fill_rate_by_policy[pol][i]:.1%}")
    print("")

fig_fill_rate, ax_fill_rate = plt.subplots(figsize=(10, 6))

for pol in policies:
    ax_fill_rate.plot(list(safety_stock_range), fill_rate_by_policy[pol], marker="o", linewidth=2, label=pol)

ax_fill_rate.set_xlabel("Safety stock (units)")
ax_fill_rate.set_ylabel("Fill rate")
ax_fill_rate.set_title("Fill rate vs. safety stock, by policy")
ax_fill_rate.yaxis.set_major_formatter(lambda y, _: f"{y:.1%}")
ax_fill_rate.grid(alpha=0.5)
ax_fill_rate.legend()

fig_fill_rate.tight_layout()
fig_fill_rate.savefig("fill_rate_by_safety_stock.png")
plt.show()

exit()


fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True)

axes = axes.flatten()

for ax, (name, df) in zip(axes, results.items()):
    ax.plot(df["Period"], df["On_Hand"], linewidth=2)
    receipt_rows = df["Receipts"] > 0
    ax.set_ylim(bottom=0)
    ax.set_title(name)
    ax.grid(alpha=0.5)

# Hide the unused plots
for ax in axes[len(results):]:
    ax.axis("off")

# Single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    bbox_to_anchor=(0.5, 1.0)
)

fig.supxlabel("Period")

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()