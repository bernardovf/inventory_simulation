#!/usr/bin/env python3
import csv
import random
import statistics

# ---- Hardcoded inputs ----
POLICY = "RsS"
re_period = 7  # review period
MOQ = 20000  # order quantity
SAFETY_STOCK = 20000  # units
PERIODS = 1000
DEMAND_MEAN = 18000
DEMAND_STD_DEV = 10000
LEAD_TIME = 31
SEED = 42
INITIAL_INVENTORY = 800000
DEMAND_DISTRIBUTION = "normal"
CSV_PATH = "simulation_output.csv"
PLOT_PATH = "simulation_plot.png"  # e.g. "simulation_plot.png"
WARMUP_PERIOD = 50

def protection_period(policy, review_period, lead_time):
    if policy in ("sQ", "sS"):
        return lead_time
    return review_period + lead_time

def reorder_point(demand_mean, protection_period_periods, safety_stock):
    return demand_mean * protection_period_periods + safety_stock

def order_up_to_level(reorder_pt, order_qty):
    return reorder_pt + order_qty

def poisson_random(lam: float) -> int:
    """Knuth's algorithm, no numpy required."""
    if lam <= 0:
        return 0
    l = pow(2.718281828459045, -lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= l:
            return k - 1

def normal_random(mean: float, std_dev: float) -> int:
    """Normal demand, rounded to integer and truncated at zero."""
    if std_dev <= 0:
        return max(0, round(mean))

    value = random.gauss(mean, std_dev)
    return max(0, round(value))

def generate_demand(demand_mean, standard_deviation, distribution):
    if distribution == "poisson":
        return poisson_random(demand_mean)
    elif distribution == "normal":
        return normal_random(demand_mean, standard_deviation)
    else:
        return demand_mean

def is_review_period(t: int, review_interval: int) -> bool:
    return t % review_interval == 0

def decide_order(policy: str, position: int, s: int, S: int, Q: int, reviewed: bool):
    """Returns (order_placed, order_qty) given the policy and current state."""
    if not reviewed:
        return False, 0
    if policy == "RS":
        return True, max(S - position, 0)
    if position > s:
        return False, 0
    if policy in ("sS", "RsS"):
        return True, max(S - position, 0)
    if policy in ("sQ", "RsQ"):
        return True, Q
    raise ValueError(f"Unknown policy: {policy}")

def simulate(policy, order_up_to, re_period, MOQ, re_point, periods, demand_mean, lead_time, initial_inventory, seed=None):
    if seed is not None:
        random.seed(seed)

    continuous = policy in ("sQ", "sS")
    review_interval = 1 if continuous else re_period

    on_hand = initial_inventory
    on_order = 0
    arrivals = {}  # period -> quantity arriving that period

    records = []

    for t in range(periods):
        arriving = arrivals.pop(t, 0)
        on_hand += arriving
        on_order -= arriving

        demand = generate_demand(DEMAND_MEAN, DEMAND_STD_DEV, "normal")

        sales = min(on_hand, demand)
        stockout_units = demand - sales
        on_hand -= sales

        position = on_hand + on_order

        reviewed = is_review_period(t, review_interval)
        order_placed, order_qty = decide_order(policy, position, re_point, order_up_to, MOQ, reviewed)

        if order_placed and order_qty > 0:
            on_order += order_qty
            arrival_t = t + max(lead_time, 1)
            arrivals[arrival_t] = arrivals.get(arrival_t, 0) + order_qty
        else:
            order_qty = 0
        records.append(
            {
                "period": t,
                "demand": demand,
                "ending_on_hand": on_hand,
                "on_order": on_order,
                "inventory_position": on_hand + on_order,
                "stockout_units": stockout_units,
                "order_placed": order_placed and order_qty > 0,
                "order_qty": order_qty,
            }
        )

    return records

def summarize(records):
    records = records[WARMUP_PERIOD:]
    total_demand = sum(r["demand"] for r in records)
    total_stockout = sum(r["stockout_units"] for r in records)
    periods_with_stockout = sum(1 for r in records if r["stockout_units"] > 0)
    avg_on_hand = statistics.mean(max(r["ending_on_hand"], 0) for r in records)
    avg_position = statistics.mean(r["inventory_position"] for r in records)
    num_orders = sum(1 for r in records if r["order_placed"])
    total_ordered = sum(r["order_qty"] for r in records)
    n = len(records)
    fill_rate = 1 - (total_stockout / total_demand) if total_demand else 1.0
    return {
        "periods": n,
        "total_demand": round(total_demand, 0),
        "total_stockout_units": round(total_stockout, 0),
        "fill_rate": fill_rate,
        "avg_on_hand_inventory": avg_on_hand,
        "num_orders_placed": num_orders,
        "total_units_ordered": total_ordered,
    }

def write_csv(records, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "period",
                "demand",
                "starting_on_hand",
                "ending_on_hand",
                "on_order",
                "inventory_position",
                "stockout_units",
                "order_placed",
                "order_qty",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r["period"],
                    r["demand"],
                    r["ending_on_hand"],
                    r["on_order"],
                    r["inventory_position"],
                    r["stockout_units"],
                    r["order_placed"],
                    r["order_qty"],
                ]
            )

def maybe_plot(records, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib to enable).")
        return

    records = records[WARMUP_PERIOD:]
    periods = [r["period"] for r in records]
    on_hand = [r["ending_on_hand"] for r in records]

    plt.figure(figsize=(10, 5))
    plt.plot(periods, on_hand, label="On-hand inventory")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Period")
    plt.ylabel("Units")
    plt.title("Inventory Simulation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(path)
    print(f"Plot saved to {path}")

def main():
    prot_period = protection_period(POLICY, re_period, LEAD_TIME)
    re_point = reorder_point(DEMAND_MEAN, prot_period, SAFETY_STOCK)
    order_up_to = order_up_to_level(re_point, MOQ)
    initial_inventory = INITIAL_INVENTORY

    records = simulate(
        policy=POLICY,
        order_up_to=order_up_to,
        re_period=re_period,
        MOQ=MOQ,
        re_point=re_point,
        periods=PERIODS,
        demand_mean=DEMAND_MEAN,
        lead_time=LEAD_TIME,
        initial_inventory=initial_inventory,
        seed=SEED,
    )

    write_csv(records, CSV_PATH)
    print(f"Wrote {len(records)} periods to {CSV_PATH}")

    summary = summarize(records)
    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if PLOT_PATH:
        maybe_plot(records, PLOT_PATH)


if __name__ == "__main__":
    main()
