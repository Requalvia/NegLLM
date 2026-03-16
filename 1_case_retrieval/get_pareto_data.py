import json
import os
import math
from typing import Optional

_PARETO_CACHE = None

def get_pareto_data(path: Optional[str] = None):
    """cache Pareto frontier data from file to memory for faster access """
    global _PARETO_CACHE
    if _PARETO_CACHE is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pareto frontier file not found: {path}")
        with open(path, "r") as f:
            _PARETO_CACHE = json.load(f)
            
    return _PARETO_CACHE

def clear_pareto_cache():
    """delete cached Pareto frontier data from memory"""
    global _PARETO_CACHE
    _PARETO_CACHE = None


# -------------------------------------------------
# pareto score calculation
# -------------------------------------------------
def pareto_score(offer, normalizer, pareto_data=None)->float:
    try:
        pareto_data = get_pareto_data()
        n_offer = normalizer(offer)

        for o in pareto_data:

            if all([n_offer[k]==o['offer'][k] for k in n_offer]):
                return o['pareto_score']
        return 0.0
    except:
        return 0.0