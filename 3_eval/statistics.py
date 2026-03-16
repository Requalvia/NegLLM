import os
import json
import numpy as np

from scipy.stats import ttest_ind
from scipy.stats import energy_distance




def calculate_social_welfare(a_vals, b_vals):

    return a_vals + b_vals





def energy_distance_nd(X, Y):
    """
    N-dimensional energy distance for vectors.
    X: (n, d)
    Y: (m, d)
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    n = X.shape[0]
    m = Y.shape[0]

    # cross distance
    cross = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)).mean()

    # within X
    dist_x = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    dist_x = dist_x[np.triu_indices(n, k=1)].mean()

    # within Y
    dist_y = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    dist_y = dist_y[np.triu_indices(m, k=1)].mean()

    return 2 * cross - dist_x - dist_y





def mean_ci_diff(x, y, alpha=0.05):

    x = np.array(x)
    y = np.array(y)

    diff = x.mean() - y.mean()

    se = np.sqrt(x.var(ddof=1)/len(x) + y.var(ddof=1)/len(y))

    from scipy.stats import t
    df = len(x) + len(y) - 2
    t_crit = t.ppf(1 - alpha/2, df)

    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    # t-test p-value
    _, p_value = ttest_ind(x, y, equal_var=False)

    return diff, ci_low, ci_high, p_value


def plot_grouped_json_scatter(
    input_dir,
    output_dir=".",
    prefix_split_key="_("  
):
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]

    groups = {}
    for fname in json_files:
        if prefix_split_key in fname:
            prefix = fname.split(prefix_split_key)[0]
        else:
            prefix = fname  
        groups.setdefault(prefix, []).append(os.path.join(input_dir, fname))

    print("\nDetected groups:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} files")



    for group_name, file_list in groups.items():





        original_path = None
        for fp in file_list:
            if "a=b=original" in fp:
                original_path = fp
                break

        if original_path is None:
            print(f"[Warning] No original file found in group {group_name}")
            continue

        with open(original_path) as f:
            orig_list = json.load(f)

        a_orig = np.array([item["a"] for item in orig_list])
        b_orig = np.array([item["b"] for item in orig_list])
        depth_orig = np.array([np.mean(item["depth"]) for item in orig_list])
        pareto_orig = np.array([np.mean(item["pareto"]) for item in orig_list])
        ab_orig_2d = np.stack([a_orig, b_orig], axis=1)

        for idx, file_path in enumerate(file_list):
            with open(file_path, "r") as f:
                data_list = json.load(f)

            a_vals = [item["a"] for item in data_list]
            b_vals = [item["b"] for item in data_list]

        text_y = 0.95

        text_y -= 0.05

        for fp in file_list:
            basename = os.path.basename(fp)
            if fp == original_path:
                continue

            with open(fp) as f:
                dat = json.load(f)

            a_vals = np.array([item["a"] for item in dat])
            b_vals = np.array([item["b"] for item in dat])
            depth_vals = np.array([np.mean(item["depth"]) for item in dat])
            pareto_vals = np.array([np.mean(item["pareto"]) for item in dat])

            ab_vals = np.stack([a_vals, b_vals], axis=1)
            ed_ab = energy_distance_nd(ab_orig_2d, ab_vals)

            sw_orig = calculate_social_welfare(a_orig, b_orig)
            sw_vals = calculate_social_welfare(a_vals, b_vals)

            diff_a, ci_a_low, ci_a_high, p_a = mean_ci_diff(a_vals, a_orig)
            diff_b, ci_b_low, ci_b_high, p_b = mean_ci_diff(b_vals, b_orig)
            diff_d, ci_d_low, ci_d_high, p_d = mean_ci_diff(depth_vals, depth_orig)
            diff_p, ci_p_low, ci_p_high, p_p = mean_ci_diff(pareto_vals, pareto_orig)

            diff_sw, ci_sw_low, ci_sw_high, p_sw = mean_ci_diff(sw_vals, sw_orig)

            diff_d = diff_d / 16.00

            stat_text = (
                f"{basename}\n"
                f"  Δa = {diff_a:.4f}  CI[{ci_a_low:.4f}, {ci_a_high:.4f}]  p={p_a:.4g}\n"
                f"  Δb = {diff_b:.4f}  CI[{ci_b_low:.4f}, {ci_b_high:.4f}]  p={p_b:.4g}\n"
                f"  Δdepth = {diff_d:.4f}  CI[{ci_d_low:.4f}, {ci_d_high:.4f}]  p={p_d:.4g}\n"
                f"  Δpareto = {diff_p:.4f}  CI[{ci_p_low:.4f}, {ci_p_high:.4f}]  p={p_p:.4g}\n"
                f"  ΔSW = {diff_sw:.4f}  CI[{ci_sw_low:.4f}, {ci_sw_high:.4f}]  p={p_sw:.4g}\n"
                f"  ED(a,b,2D) = {ed_ab:.4f}"
            )

            print("--------------------------------------------------")
            print(stat_text)
            print("--------------------------------------------------")








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PG-MCTS of negotiation trees")


    parser.add_argument('--dir', type=str, help='which dir for statistics',  required=True)

    args = parser.parse_args()

    if 'rq1' in args.dir or 'rq2' in args.dir:
        input_dir = args.dir
        output_dir = input_dir
        plot_grouped_json_scatter(input_dir, output_dir)
    elif 'rq3' in args.dir:
        from rq3 import check
        check(args.dir)
    else:
        print("Unknown directory type for statistics.")
