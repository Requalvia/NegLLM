# -*- coding: utf-8 -*-
import numpy as np

import os, json
from pathlib import Path

def reward_normalize(reward):
    if not isinstance(reward, (int, float)):
        reward = reward[0]
    return reward


def offer_normalize(offer):
    return offer

# ====== Global Preferences ======
# agent，
global_preference_a = None
global_preference_b = None

# ====== Utility Functions ======

def compute_issue_score(option: str, pref_list: list) -> float:
    """
    ：
    1=1, 2=2/3, 3=1/3, =0
    """
    if option == pref_list[0]:
        return 1.0
    elif option == pref_list[1]:
        return 2.0/3.0
    elif option == pref_list[2]:
        return 1.0/3.0
    else:
        return 0.0


def uf_a(offer: dict):
    """
    Agent A 。
    : A=3, B=2, C=1
    : 1, 2/3, 1/3
     reward  [0, 1]
    """
    global global_preference_a
    if global_preference_a is None:
        pth = Path(os.path.dirname(__file__)) / 'pref_a.json'
        with open(pth) as jf:
            global_preference_a = json.load(jf)

    try:
        reward = 0
        total_weight = sum(v["weight"] for v in global_preference_a.values())
        explanation_parts = []

        for issue, meta in global_preference_a.items():
            weight = meta["weight"]
            pref_list = meta["options"]
            chosen = offer.get(issue)

            if chosen is None:
                continue

            # compute score (rank-based)
            if chosen in pref_list:
                rank = pref_list.index(chosen)
                score = 1 - (rank / (len(pref_list) - 1))   # 1, 2/3, 1/3
            else:
                rank = "?"
                score = 0

            weighted = score * weight
            reward += weighted

            explanation_parts.append(
                f"{issue}: chose {chosen}, rank={rank}, "
                f"base={score:.2f}, weighted={weighted:.2f}"
            )

        reward_norm = reward / total_weight
        explanation = " ; ".join(explanation_parts)
        return reward_norm, explanation

    except Exception as e:
        print("UF ERROR:", e)
        print("Offer:", offer)
        return 0, "Illegal offer"


def uf_b(offer: dict):
    """
    Agent B 。
     uf_a ， global_preference_b。
    """
    global global_preference_b
    if global_preference_b is None:
        pth = Path(os.path.dirname(__file__)) / "pref_b.json"
        with open(pth) as jf:
            global_preference_b = json.load(jf)

    try:
        reward = 0
        total_weight = sum(v["weight"] for v in global_preference_b.values())
        explanation_parts = []

        for issue, meta in global_preference_b.items():
            weight = meta["weight"]
            pref_list = meta["options"]
            chosen = offer.get(issue)

            if chosen is None:
                continue

            # compute score (rank-based)
            if chosen in pref_list:
                rank = pref_list.index(chosen)
                score = 1 - (rank / (len(pref_list) - 1))  # 1, 2/3, 1/3
            else:
                rank = "?"
                score = 0

            weighted = score * weight
            reward += weighted

            explanation_parts.append(
                f"{issue}: chose {chosen}, rank={rank}, "
                f"base={score:.2f}, weighted={weighted:.2f}"
            )

        reward_norm = reward / total_weight
        explanation = " ; ".join(explanation_parts)
        return reward_norm, explanation

    except Exception as e:
        print("UF ERROR:", e)
        print("Offer:", offer)
        return 0, "Illegal offer"
