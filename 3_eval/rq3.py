


import json
import glob
import numpy as np
import re
from collections import defaultdict

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_filename(name):
    """
    env: antique / citymanagement / travel
    mapping: dict, e.g. {"a": "ft", "b": "original"}
    """
    env = re.search(r"env_([a-zA-Z]+)_", name).group(1)
    role = re.search(r"\((.*?)\)", name).group(1)
    mapping = dict(x.split("=") for x in role.split(","))
    return env, mapping

def mean_std(x):
    return np.mean(x), np.std(x, ddof=1)


def check(root):

    stats = defaultdict(lambda: {"ft": [], "original": []})

    for path in glob.glob("*.json", root_dir=root):
        env, mapping = parse_filename(path)

        data = load_json(root+'/'+path)

        for item in data:
            for agent in ["a", "b"]:
                model_type = mapping[agent]   
                stats[env][model_type].append(item[agent])

    for env in sorted(stats.keys()):
        print(f"Env: {env}")

        ft_mean, ft_std = mean_std(stats[env]["ft"])
        ori_mean, ori_std = mean_std(stats[env]["original"])

        print(f"  ft       : mean = {ft_mean:.4f}, std = {ft_std:.2f}")
        print(f"  original : mean = {ori_mean:.4f}, std = {ori_std:.2f}")
        print("-" * 60)
