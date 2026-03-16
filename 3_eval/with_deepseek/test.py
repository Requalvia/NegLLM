# import json
# import glob
# import numpy as np
# import os

# def stat_file(path):
#     with open(path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     a_vals = [item["a"] for item in data]
#     b_vals = [item["b"] for item in data]

#     a_mean = np.mean(a_vals)
#     a_std  = np.std(a_vals, ddof=1)  # 
#     b_mean = np.mean(b_vals)
#     b_std  = np.std(b_vals, ddof=1)

#     return a_mean, a_std, b_mean, b_std


# if __name__ == "__main__":
#     json_files = sorted(glob.glob("*.json"))

#     for path in json_files:
#         a_mean, a_std, b_mean, b_std = stat_file(path)

#         print(f"File: {os.path.basename(path)}")
#         print(f"  a: mean = {a_mean:.6f}, std = {a_std:.6f}")
#         print(f"  b: mean = {b_mean:.6f}, std = {b_std:.6f}")
#         print("-" * 60)


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
    :
    env: antique / citymanagement / travel
    mapping: dict, e.g. {"a": "ft", "b": "original"}
    """
    env = re.search(r"env_([a-zA-Z]+)_", name).group(1)
    role = re.search(r"\((.*?)\)", name).group(1)
    mapping = dict(x.split("=") for x in role.split(","))
    return env, mapping

def mean_std(x):
    return np.mean(x), np.std(x, ddof=1)

if __name__ == "__main__":
    # env -> {"ft": [...], "original": [...]}
    stats = defaultdict(lambda: {"ft": [], "original": []})

    for path in glob.glob("*.json"):
        env, mapping = parse_filename(path)
        data = load_json(path)

        for item in data:
            for agent in ["a", "b"]:
                model_type = mapping[agent]   # ft or original
                stats[env][model_type].append(item[agent])

    # 
    for env in sorted(stats.keys()):
        print(f"Env: {env}")

        ft_mean, ft_std = mean_std(stats[env]["ft"])
        ori_mean, ori_std = mean_std(stats[env]["original"])

        print(f"  ft       : mean = {ft_mean:.6f}, std = {ft_std:.6f}")
        print(f"  original : mean = {ori_mean:.6f}, std = {ori_std:.6f}")
        print("-" * 60)
