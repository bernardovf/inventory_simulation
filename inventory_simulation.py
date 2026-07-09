#!/usr/bin/env python3
"""Basic single-item inventory simulation.

Inputs:
    S  - order up-to level
    R  - review period (periods between reviews; R=1 means continuous/every-period review)
    Q  - order quantity (used by the fixed-quantity policies)
    s  - reorder point

Supported policies (pick with --policy):
    sQ   - continuous review: every period, if inventory position <= s, order a fixed quantity Q
    sS   - continuous review: every period, if inventory position <= s, order up to S
    RS   - periodic review:   every R periods, order up to S unconditionally
    RsS  - periodic review:   every R periods, if inventory position <= s, order up to S
    RsQ  - periodic review:   every R periods, if inventory position <= s, order a fixed quantity Q

Demand is generated per period from a Poisson distribution (mean --demand-mean).
Orders arrive --lead-time periods after being placed. Unmet demand is backordered
(on-hand inventory can go negative) and filled once stock arrives.
"""
import argparse
import csv
import random
import statistics
from dataclasses import dataclass, field


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


@dataclass
class PeriodRecord:
    period: int
    demand: int
    starting_on_hand: int
    ending_on_hand: int
    on_order: int
    inventory_position: int
    stockout_units: int
    order_placed: bool
    order_qty: int


@dataclass
class SimulationResult:
    records: list = field(default_factory=list)

    def summary(self):
        total_demand = sum(r.demand for r in self.records)
        total_stockout = sum(r.stockout_units for r in self.records)
        periods_with_stockout = sum(1 for r in self.records if r.stockout_units > 0)
        avg_on_hand = statistics.mean(max(r.ending_on_hand, 0) for r in self.records)
        avg_position = statistics.mean(r.inventory_position for r in self.records)
        num_orders = sum(1 for r in self.records if r.order_placed)
        total_ordered = sum(r.order_qty for r in self.records)
        n = len(self.records)
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


def is_review_period(t: int, R: int) -> bool:
    return t % R == 0


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


def simulate(
    policy: str,
    S: int,
    R: int,
    Q: int,
    s: int,
    periods: int,
    demand_mean: float,
    lead_time: int,
    initial_inventory: int,
    seed: int = None,
) -> SimulationResult:
    if seed is not None:
        random.seed(seed)

    continuous = policy in ("sQ", "sS")
    review_interval = 1 if continuous else R

    on_hand = initial_inventory
    on_order = 0
    arrivals = {}  # period -> quantity arriving that period

    result = SimulationResult()

    for t in range(periods):
        arriving = arrivals.pop(t, 0)
        on_hand += arriving
        on_order -= arriving

        demand = poisson_random(demand_mean)
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

        result.records.append(
            PeriodRecord(
                period=t,
                demand=demand,
                starting_on_hand=starting_on_hand,
                ending_on_hand=on_hand,
                on_order=on_order,
                inventory_position=on_hand + on_order,
                stockout_units=stockout_units,
                order_placed=order_placed and order_qty > 0,
                order_qty=order_qty,
            )
        )

    return result


def write_csv(result: SimulationResult, path: str):
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
        for r in result.records:
            writer.writerow(
                [
                    r.period,
                    r.demand,
                    r.starting_on_hand,
                    r.ending_on_hand,
                    r.on_order,
                    r.inventory_position,
                    r.stockout_units,
                    r.order_placed,
                    r.order_qty,
                ]
            )


def maybe_plot(result: SimulationResult, path: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install matplotlib to enable).")
        return

    periods = [r.period for r in result.records]
    on_hand = [r.ending_on_hand for r in result.records]
    position = [r.inventory_position for r in result.records]

    plt.figure(figsize=(10, 5))
    plt.plot(periods, on_hand, label="On-hand inventory")
    plt.plot(periods, position, label="Inventory position", linestyle="--")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.xlabel("Period")
    plt.ylabel("Units")
    plt.title("Inventory Simulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    print(f"Plot saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="Basic inventory simulation (S, R, Q, s).")
    parser.add_argument("--policy", choices=["sQ", "sS", "RS", "RsS", "RsQ"], default="RsS")
    parser.add_argument("--S", type=int, default=100, help="Order up-to level")
    parser.add_argument("--R", type=int, default=7, help="Review period")
    parser.add_argument("--Q", type=int, default=50, help="Order quantity")
    parser.add_argument("--s", type=int, default=30, help="Reorder point")
    parser.add_argument("--periods", type=int, default=100, help="Number of periods to simulate")
    parser.add_argument("--demand-mean", type=float, default=10.0, help="Mean demand per period")
    parser.add_argument("--lead-time", type=int, default=3, help="Lead time in periods")
    parser.add_argument("--initial-inventory", type=int, default=None, help="Starting on-hand inventory (defaults to S)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--csv", type=str, default="simulation_output.csv", help="Path to write CSV output")
    parser.add_argument("--plot", type=str, default=None, help="Path to save a plot (requires matplotlib)")
    args = parser.parse_args()

    initial_inventory = args.initial_inventory if args.initial_inventory is not None else args.S

    result = simulate(
        policy=args.policy,
        S=args.S,
        R=args.R,
        Q=args.Q,
        s=args.s,
        periods=args.periods,
        demand_mean=args.demand_mean,
        lead_time=args.lead_time,
        initial_inventory=initial_inventory,
        seed=args.seed,
    )

    write_csv(result, args.csv)
    print(f"Wrote {len(result.records)} periods to {args.csv}")

    summary = result.summary()
    print("\nSummary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.plot:
        maybe_plot(result, args.plot)


if __name__ == "__main__":
    main()
