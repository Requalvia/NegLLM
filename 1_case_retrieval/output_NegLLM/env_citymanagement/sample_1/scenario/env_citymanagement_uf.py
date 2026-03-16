import math


alpha = 0.06
beta = 0.9


def reward_normalize(reward):

    if isinstance(reward, dict):
        reward = list(reward.values())[0]
    if not isinstance(reward, (int, float)):
        reward = reward[0]
    return reward



def offer_normalize(offer):

    return offer

def uf_a(offer: dict):
    """
    Mayor's utility:
    U = mean(v) - alpha * std(v)
    Explanation lists each component and how mean/std are computed.
    """
    I = offer["Industrial"]
    S = offer["Services"]
    W = offer["Welfare"]
    T = offer["Transport"]

    values = [I, S, W, T]

    mean_v = sum(values) / 4
    var_v = sum((v - mean_v) ** 2 for v in values) / 4
    std_v = math.sqrt(var_v)

    reward = 2*mean_v - alpha * std_v

    explanation = (
        f"Values={values}, mean={mean_v:.2f}, std={std_v:.2f}; "
        f"Total = mean - alpha*std = {mean_v:.2f} - {alpha}*{std_v:.2f} = {reward:.2f}"
    )


    reward = (reward - 2.0) / 18.0
    explanation += f" normalized:{reward}"

    return reward, explanation



def uf_b(offer: dict):
    """
    Developer's utility:
    U = 2*Industrial + 2*Services - beta*(Welfare + Transport)
    Explanation lists each component contribution.
    """
    I = offer["Industrial"]
    S = offer["Services"]
    W = offer["Welfare"]
    T = offer["Transport"]

    values = [I, S, W, T]

    part_I =  I
    part_S =  S
    penalty = beta * (W + T)


    reward = part_I + part_S - penalty

    explanation_parts = [
        f"Industrial: {I} = {part_I}",
        f"Services: {S} = {part_S}",
        f"Welfare+Transport penalty: -beta*({W}+{T}) = -{beta}*{W+T} = -{penalty}"
    ]

    explanation = " ; ".join(explanation_parts) + f" ; Total = {reward}"



    reward = (reward + 18.0) / 36.0
    explanation += f" normalized:{reward}"

    return reward, explanation
