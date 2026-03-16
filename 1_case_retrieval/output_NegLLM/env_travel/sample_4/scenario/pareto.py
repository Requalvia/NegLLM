import itertools
import json
import math
from tqdm import tqdm

from env_travel_uf import uf_a, uf_b


# -------------------------------
# 2. 
# -------------------------------
from itertools import product
import json
import os
from pathlib import Path

def load_preferences(pref_path):
    with open(pref_path) as jf:
        return json.load(jf)

def build_offer_space():
    """
     pref_a.json（ pref_b.json） offer 。
    ：
    {
        "Sites": {"weight": 2, "options": [...]},
        "Amusement": {"weight": 1, "options": [...]},
        "Meals": {"weight": 3, "options": [...]}
    }
    """
    #  A ， 3 options
    pth = Path(os.path.dirname(__file__)) / "pref_a.json"
    pref = load_preferences(pth)

    issues = list(pref.keys())
    option_lists = [pref[iss]["options"] for iss in issues]

    # 
    all_combinations = list(product(*option_lists))

    #  offer 
    offers = []
    for combo in all_combinations:
        offer = {issue: choice for issue, choice in zip(issues, combo)}
        offers.append(offer)

    return offers


# -------------------------------
# 3. 
# -------------------------------
def dominates(r1, r2):
    """
    r1 dominates r2 if r1 is at least as good in all reward dimensions
    and strictly better in at least one.
    """
    keys = ["a", "b"]
    better_or_equal = all(r1[k] >= r2[k] for k in keys)
    strictly_better = any(r1[k] > r2[k] for k in keys)
    return better_or_equal and strictly_better

# -------------------------------
# 4. 
# -------------------------------
def compute_pareto_frontier(uf_a, uf_b, offer_space):
    pareto_front = []

    for offer in tqdm(offer_space, desc="Computing Pareto frontier"):
        a, _ = uf_a(offer)
        b, _ = uf_b(offer)

        point = {
            "offer": offer,
            "a": a,
            "b": b
        }

        # 
        if not any(dominates(existing, {"a": a, "b": b})
                   for existing in pareto_front):

            # 
            pareto_front = [
                p for p in pareto_front
                if not dominates({"a": a, "b": b}, p)
            ]

            pareto_front.append(point)

    return pareto_front

# -------------------------------
# 5.  offer 
# -------------------------------
def compute_pareto_distances(all_offers, pareto_front):
    distances = []

    for o in all_offers:
        # 
        if any(o['offer'] == p['offer'] for p in pareto_front):
            dist = 0.0
        else:
            dist = min(
                math.sqrt((o['a'] - p['a'])**2 + (o['b'] - p['b'])**2)
                for p in pareto_front
            )

        o['pareto_distance'] = dist
        distances.append(dist)

    # 
    d_min, d_max = min(distances), max(distances)
    for o in all_offers:
        if d_max > d_min:
            o['pareto_distance_norm'] = (o['pareto_distance'] - d_min) / (d_max - d_min)
            o['pareto_score'] = 1.0 - o['pareto_distance_norm']
        else:
            o['pareto_distance_norm'] = 0.0
            o['pareto_score'] = 1.0

    return all_offers

# def compute_pareto_distances(all_offers, pareto_front):
#     distances = []
#     for offer in all_offers:
#         if any(offer['offer'] == p['offer'] for p in pareto_front):
#             dist = 0.0
#         else:
#             dist = min(
#                 math.sqrt(
#                     (offer['a'] - p['a'])**2 +
#                     (offer['b'] - p['b'])**2
#                 ) for p in pareto_front
#             )
#         offer['pareto_distance'] = dist
#         distances.append(dist)

#     #  [0,1]
#     d_min, d_max = min(distances), max(distances)
#     for offer in all_offers:
#         if d_max > d_min:
#             offer['pareto_distance_norm'] = (offer['pareto_distance'] - d_min) / (d_max - d_min)
#             # ，：
#             offer['pareto_score'] = 1.0 - offer['pareto_distance_norm']
#         else:
#             offer['pareto_distance_norm'] = 0.0
#             offer['pareto_score'] = 1.0

#     return all_offers




import matplotlib.pyplot as plt

# -------------------------------
# 7.  offer
# -------------------------------
def plot_pareto(all_offers, pareto_front, save_path="pareto_plot.png"):
    fig, ax = plt.subplots(figsize=(8,6))

    #  offer
    sellers = [o['a'] for o in all_offers]
    bs = [o['b'] for o in all_offers]
    ax.scatter(sellers, bs, color='lightgray', alpha=0.5, label='All offers')

    # 
    pf_seller = [p['a'] for p in pareto_front]
    pf_buyer = [p['b'] for p in pareto_front]
    #  a 
    sorted_pf = sorted(zip(pf_seller, pf_buyer))
    pf_seller_sorted, pf_buyer_sorted = zip(*sorted_pf)
    ax.plot(pf_seller_sorted, pf_buyer_sorted, color='red', marker='o', linestyle='-', linewidth=2, label='Pareto frontier')

    ax.set_xlabel('Seller reward')
    ax.set_ylabel('Buyer reward')
    ax.set_title('Pareto Frontier for Antique Offers')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Pareto plot saved to {save_path}")

# -------------------------------
# 6. 
# -------------------------------
# if __name__ == "__main__":
def main():
    # 1.  possible offer (27 )
    offer_space = build_offer_space()

    # pth = Path(os.path.dirname(__file__)) / "pref_a.json"
    # pref = load_preferences(pth)

    # 2. 
    pareto_front = compute_pareto_frontier(uf_a, uf_b, offer_space)
    print(f"Pareto frontier has {len(pareto_front)} points.")

    # 3.  offer reward
    all_offers = []
    for offer in offer_space:
        a, _ = uf_a(offer)
        b, _ = uf_b(offer)
        all_offers.append({"offer": offer, "a": a, "b": b})

    # 4.  + 
    all_offers = compute_pareto_distances(all_offers, pareto_front)

    import os
    figpath = Path(os.path.dirname(__file__)) / "pareto.png"
    jsonpath = Path(os.path.dirname(__file__)) / "pareto.json"

    # 5. 
    with open(jsonpath, "w") as f:
        json.dump(all_offers, f, indent=2)

    # 6. 
    plot_pareto(all_offers, pareto_front, figpath)
