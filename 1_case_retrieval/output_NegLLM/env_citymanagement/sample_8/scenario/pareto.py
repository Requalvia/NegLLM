import itertools
import json
import math
from tqdm import tqdm
from env_citymanagement_uf import uf_a, uf_b
import matplotlib.pyplot as plt


def build_issue_space():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return values, values, values, values



def dominates(r1, r2):
    keys = ["na", "nb"]
    better_or_equal = all(r1[k] >= r2[k] for k in keys)
    strictly_better = any(r1[k] > r2[k] for k in keys)
    return better_or_equal and strictly_better



def compute_all_rewards(values_I, values_S, values_W, values_T):
    offers = []
    for I, S, W, T in itertools.product(values_I, values_S, values_W, values_T):
        offer = {
            "Industrial": I,
            "Services": S,
            "Welfare": W,
            "Transport": T
        }
        a, _ = uf_a(offer)
        b, _ = uf_b(offer)
        offers.append({"offer": offer, "a": a, "b": b})

    return offers



def normalize_rewards(all_offers):
    a_vals = [o["a"] for o in all_offers]
    b_vals = [o["b"] for o in all_offers]
    a_min, a_max = min(a_vals), max(a_vals)
    b_min, b_max = min(b_vals), max(b_vals)
    print(    a_min, a_max ,b_min, b_max )

    for o in all_offers:
        o["na"] = (o["a"] - a_min) / (a_max - a_min) 
        o["nb"] = (o["b"] - b_min) / (b_max - b_min) 

    return all_offers


def compute_pareto_frontier_normalized(all_offers):
    pareto_front = []

    for o in tqdm(all_offers, desc="Computing Pareto frontier (normalized)"):
        na, nb = o["na"], o["nb"]

        if not any(dominates(existing, {"na": na, "nb": nb}) for existing in pareto_front):
            pareto_front = [
                p for p in pareto_front
                if not dominates({"na": na, "nb": nb}, p)
            ]
            pareto_front.append(o)

    return pareto_front


def compute_pareto_distances(all_offers, pareto_front):
    distances = []

    for o in all_offers:
        if any(o["offer"] == p["offer"] for p in pareto_front):
            dist = 0.0
        else:
            dist = min(
                math.sqrt((o["na"] - p["na"]) ** 2 + (o["nb"] - p["nb"]) ** 2)
                for p in pareto_front
            )
        o["pareto_distance"] = dist
        distances.append(dist)


    d_min, d_max = min(distances), max(distances)
    for o in all_offers:
        if d_max > d_min:
            o["pareto_distance_norm"] = (o["pareto_distance"] - d_min) / (d_max - d_min)
            o["pareto_score"] = 1.0 - o["pareto_distance_norm"]
        else:
            o["pareto_distance_norm"] = 0.0
            o["pareto_score"] = 1.0

    return all_offers



def plot_pareto(all_offers, pareto_front, save_path="pareto_city_normalized.png"):
    fig, ax = plt.subplots(figsize=(8,6))


    na_vals = [o["na"] for o in all_offers]
    nb_vals = [o["nb"] for o in all_offers]
    ax.scatter(na_vals, nb_vals, color='lightgray', alpha=0.5, label="All offers")


    pf_na = [p["na"] for p in pareto_front]
    pf_nb = [p["nb"] for p in pareto_front]


    pf_na_sorted, pf_nb_sorted = zip(*sorted(zip(pf_na, pf_nb)))

    ax.plot(pf_na_sorted, pf_nb_sorted, color="red", marker="o",
            linewidth=2, label="Pareto frontier (normalized)")

    ax.set_xlabel("Normalized Mayor reward (na)")
    ax.set_ylabel("Normalized Developer reward (nb)")
    ax.set_title("Pareto Frontier (Normalized Reward Space)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Normalized Pareto plot saved to {save_path}")



def main():
    values_I, values_S, values_W, values_T = build_issue_space()

    all_offers = compute_all_rewards(values_I, values_S, values_W, values_T)
    all_offers = normalize_rewards(all_offers)

    pareto_front = compute_pareto_frontier_normalized(all_offers)
    print(f"Calculated Pareto frontier with {len(pareto_front)} points.")

    all_offers = compute_pareto_distances(all_offers, pareto_front)

    import os
    from pathlib import Path

    figpath = Path(os.path.dirname(__file__)) / "pareto.png"
    jsonpath = Path(os.path.dirname(__file__)) / "pareto.json"


    with open(jsonpath, "w") as f:
        json.dump(all_offers, f, indent=2)


    plot_pareto(all_offers, pareto_front, figpath)


