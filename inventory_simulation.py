import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_inventory(demand, policy, lead_time, initial_on_hand, safety_stock, MOQ=None, Review_Period=None):
    valid_policies = {
        "(s,Q)",
        "(R,S)",
        "(s,S)",
        "(R,s,S)",
        "(R,s,Q)"}

    demand = np.asarray(demand, dtype=float)
    periods = len(demand)

    # Extra space is needed for orders arriving after the simulation horizon
    scheduled_receipts = np.zeros(periods + lead_time + 1, dtype=float)

    on_hand = np.zeros(periods)
    receipts = np.zeros(periods)
    order_qty = np.zeros(periods)
    fulfilled_demand = np.zeros(periods)
    lost_sales = np.zeros(periods)

    inventory_position_before_order = np.zeros(periods)
    inventory_position_after_order = np.zeros(periods)
    reorder_point = np.zeros(periods)
    order_up_to_level = np.zeros(periods)

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
        # actual future demand rather than the series-wide average.
        future_demand = demand[t + 1:]
        demand_lead_time = future_demand[:lead_time].sum()
        demand_protection_period = future_demand[:lead_time + Review_Period].sum()

        s = demand_lead_time + safety_stock
        S = demand_protection_period + safety_stock

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
            arrival_period = t + lead_time

            if arrival_period < len(scheduled_receipts):
                scheduled_receipts[arrival_period] += qty

        inventory_position_after_order[t] = ip + qty

    results = pd.DataFrame({
        "Period": np.arange(periods),
        "Demand": demand,
        "Receipts": receipts,
        "Fulfilled_Demand": fulfilled_demand,
        "Lost_Sales": lost_sales,
        "On_Hand": on_hand,
        "Order_Qty": order_qty,
        "Reorder_Point": reorder_point,
        "Order_Up_To_Level": order_up_to_level,
        "Inventory_Position_Before_Order": inventory_position_before_order,
        "Inventory_Position_After_Order": inventory_position_after_order
    })

    return results

rng = np.random.default_rng(42)

time = 500
warm_up_period = 60
demand_average = 100
demand_std_deviation = 75

demand = np.maximum(rng.normal(demand_average, demand_std_deviation, time).round(),0)
init_on_hand = 100
average_lead_time = 8
MOQ = 500
review_period = 1
safety_stock_units = 0

policies = ["(C,MOQ)", "(C,No MOQ)", "(P,MOQ)", "(P,Non MOQ)"]

results = {}

for safety_stock_units in range(50, 750, 50):
    for pol in policies:
        results_pol = simulate_inventory(
            demand=demand,
            policy=pol,
            lead_time=average_lead_time,
            initial_on_hand=init_on_hand,
            safety_stock=safety_stock_units,
            MOQ=MOQ,
            Review_Period=review_period)

        results_pol = results_pol[results_pol["Period"] > warm_up_period]
        total_demand = results_pol["Demand"].sum()
        total_fulfilled = results_pol["Fulfilled_Demand"].sum()
        fill_rate = (total_fulfilled / total_demand
            if total_demand > 0
            else float("nan"))
        print(f"{safety_stock_units}, {pol}: {fill_rate:.1%}")
        results[pol] = results_pol
    print("")

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