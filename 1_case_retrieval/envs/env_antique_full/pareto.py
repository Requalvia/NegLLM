import itertools
import json
import math
from tqdm import tqdm
from env_antique_full_uf import uf_a, uf_b 




# def build_price_space(coin_range=(600, 1350), painting_range=(150, 900), step=5):
def build_price_space(coin_range=(400, 800), painting_range=(200, 600), step=5):
    coin_prices = list(range(coin_range[0], coin_range[1] + 1, step))
    painting_prices = list(range(painting_range[0], painting_range[1] + 1, step))
    return coin_prices, painting_prices


def dominates(r1, r2):
    """
    r1 dominates r2 if r1 is at least as good in all reward dimensions
    and strictly better in at least one.
    """
    keys = ["a", "b"]
    better_or_equal = all(r1[k] >= r2[k] for k in keys)
    strictly_better = any(r1[k] > r2[k] for k in keys)
    return better_or_equal and strictly_better


def compute_pareto_frontier(uf_seller, uf_buyer, coin_prices, painting_prices):
    pareto_front = []
    all_combinations = list(itertools.product(coin_prices, painting_prices))
    
    for coin, painting in tqdm(all_combinations, desc="Computing Pareto frontier"):
        offer = {"Coin": coin, "Painting": painting}


        a, _ = uf_seller(offer)
        b, _ = uf_buyer(offer)

        point = {
            "offer": offer,
            "a": a,
            "b": b
        }


        if not any(dominates(existing, {"a": a, "b": b})
                   for existing in pareto_front):

            pareto_front = [p for p in pareto_front
                            if not dominates({"a": a, "b": b}, p)]
            pareto_front.append(point)
    
    return pareto_front


def compute_pareto_distances(all_offers, pareto_front):
    distances = []
    for offer in all_offers:
        if any(offer['offer'] == p['offer'] for p in pareto_front):
            dist = 0.0
        else:
            dist = min(
                math.sqrt(
                    (offer['a'] - p['a'])**2 +
                    (offer['b'] - p['b'])**2
                ) for p in pareto_front
            )
        offer['pareto_distance'] = dist
        distances.append(dist)

    # normalize to [0,1]
    d_min, d_max = min(distances), max(distances)
    for offer in all_offers:
        if d_max > d_min:
            offer['pareto_distance_norm'] = (offer['pareto_distance'] - d_min) / (d_max - d_min)

            offer['pareto_score'] = 1.0 - offer['pareto_distance_norm']
        else:
            offer['pareto_distance_norm'] = 0.0
            offer['pareto_score'] = 1.0

    return all_offers




import matplotlib.pyplot as plt


def plot_pareto(all_offers, pareto_front, save_path="pareto_plot.png"):
    fig, ax = plt.subplots(figsize=(8,6))


    sellers = [o['a'] for o in all_offers]
    bs = [o['b'] for o in all_offers]
    ax.scatter(sellers, bs, color='lightgray', alpha=0.5, label='All offers')

    pf_seller = [p['a'] for p in pareto_front]
    pf_buyer = [p['b'] for p in pareto_front]

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


if __name__ == "__main__":

    coin_prices, painting_prices = build_price_space(step=5)


    pareto_front = compute_pareto_frontier(uf_a, uf_b, coin_prices, painting_prices)
    print(f"Calculated Pareto frontier with {len(pareto_front)} points.")


    all_offers = []
    for coin, painting in itertools.product(coin_prices, painting_prices):
        offer = {"Coin": coin, "Painting": painting}

        a, _ = uf_a(offer)
        b, _ = uf_b(offer)
        all_offers.append({
            "offer": offer,
            "a": a,
            "b": b
        })


    all_offers = compute_pareto_distances(all_offers, pareto_front)


    with open("pareto.json", "w") as f:
        json.dump(all_offers, f, indent=2)

    print(f"Saved {len(all_offers)} offers with Pareto distances to 'pareto_frontier_with_distances.json'.")

    plot_pareto(all_offers, pareto_front)

