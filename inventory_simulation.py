#!/usr/bin/env python3
"""Basic single-item inventory simulation.

Inputs:
    R              - review period (periods between reviews; R=1 means continuous/every-period review)
    Q              - order quantity (used by the fixed-quantity policies)
    SAFETY_STOCK   - safety stock, in units
    S, s           - order up-to level and reorder point; computed below from demand,
                     lead time, review period, and safety stock (see reorder_point/order_up_to_level)

Supported policies (set POLICY below):
    sQ   - continuous review: every period, if inventory position <= s, order a fixed quantity Q
    sS   - continuous review: every period, if inventory position <= s, order up to S
    RS   - periodic review:   every R periods, order up to S unconditionally
    RsS  - periodic review:   every R periods, if inventory position <= s, order up to S
    RsQ  - periodic review:   every R periods, if inventory position <= s, order a fixed quantity Q

Reorder point / order-up-to level:
    The reorder point must cover expected demand over the "protection period" - the
    time between placing an order and being able to react to the next one - plus a
    safety stock buffer:
        s = demand_mean * protection_period + safety_stock
    For continuous review (sQ, sS) the protection period is just the lead time L,
    since inventory position is checked every period. For periodic review (RS, RsS,
    RsQ) it's R + L, since a stockout risk window also includes waiting for the next
    review. The order-up-to level S is set to cover the same protection period plus
    one order batch: S = s + Q.

Demand is generated per period from a Poisson distribution (mean DEMAND_MEAN).
Orders arrive LEAD_TIME periods after being placed. Unmet demand is backordered
(on-hand inventory can go negative) and filled once stock arrives.
"""
import csv
import random
import statistics

# ---- Hardcoded inputs ----
POLICY = "RsS"
R = 7  # review period
Q = 50  # order quantity
SAFETY_STOCK = 20  # units
PERIODS = 60
DEMAND_MEAN = 10.0
LEAD_TIME = 3
SEED = 42
CSV_PATH = "simulation_output.csv"
PLOT_PATH = "simulation_plot.png"  # e.g. "simulation_plot.png"


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

def generate_demand(demand_mean, standard_deviation, distribution):
    if distribution == "poisson":
        return poisson_random(demand_mean)
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

def simulate(policy, S, R, Q, s, periods, demand_mean, lead_time, initial_inventory, seed=None):
    if seed is not None:
        random.seed(seed)

    continuous = policy in ("sQ", "sS")
    review_interval = 1 if continuous else R

    on_hand = initial_inventory
    on_order = 0
    arrivals = {}  # period -> quantity arriving that period

    records = []

    for t in range(periods):
        arriving = arrivals.pop(t, 0)
        on_hand += arriving
        on_order -= arriving

        demand = generate_demand(demand_mean, 0, "constant")
        starting_on_hand = on_hand
        on_hand -= demand
        stockout_units = max(demand - max(starting_on_hand, 0), 0)

        position = on_hand + on_order
        reviewed = is_review_period(t, review_interval)
        order_placed, order_qty = decide_order(policy, position, s, S, Q, reviewed)

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
                "starting_on_hand": starting_on_hand,
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
    total_demand = sum(r["demand"] for r in records)
    total_stockout = sum(r["stockout_units"] for r in records)
    periods_with_stockout = sum(1 for r in records if r["stockout_units"] > 0)
    avg_on_hand = statistics.mean(max(r["ending_on_hand"], 0) for r in records)
    avg_position = statistics.mean(r["inventory_position"] for r in records)
    num_orders = sum(1 for r in records if r["order_placed"])
    total_ordered = sum(r["order_qty"] for r in records)
    n = len(records)
    fill_rate = 1 - (total_stockout / total_demand) if total_demand else 1.0
    cycle_service_level = 1 - (periods_with_stockout / n) if n else 1.0
    return {
        "periods": n,
        "total_demand": total_demand,
        "total_stockout_units": total_stockout,
        "fill_rate": fill_rate,
        "cycle_service_level": cycle_service_level,
        "avg_on_hand_inventory": avg_on_hand,
        "avg_inventory_position": avg_position,
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
                    r["starting_on_hand"],
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

    periods = [r["period"] for r in records]
    on_hand = [r["ending_on_hand"] for r in records]
    position = [r["inventory_position"] for r in records]

    plt.figure(figsize=(10, 5))
    plt.plot(periods, on_hand, label="On-hand inventory")
    plt.plot(periods, position, label="Inventory position", linestyle="--")
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
    pp = protection_period(POLICY, R, LEAD_TIME)
    s = reorder_point(DEMAND_MEAN, pp, SAFETY_STOCK)
    S = order_up_to_level(s, Q)
    initial_inventory = S

    print(f"Policy: {POLICY}  protection period: {pp}  reorder point s: {s:.1f}  order-up-to S: {S:.1f}")

    records = simulate(
        policy=POLICY,
        S=S,
        R=R,
        Q=Q,
        s=s,
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
