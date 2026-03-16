import json, os
import pathlib
from typing import List, Dict, Any
import importlib.util
import math
import numpy as np
import glob
from tqdm import tqdm

def watch_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.mean(data['total_value'][-10:])

def print_tree_structure(
    node,
    main_viewer='a',
    prefix: str = "",
    is_last: bool = True,
    depth: int = 0,
    max_depth: int = 10
):
    """
    Print the JSON tree structure (with tree symbols ├── │ └──).
    Also display:
    - u: Current immediate reward
    - r: Cumulative reward (expected_reward)
    - f: Reward including future expected reward (expected_with_future)
    """



    if depth > max_depth:
        return

    role = node.get("role", "?")
    alp = float(node.get("ALP", 0.0) or 0.0)
    offer = node.get("offer", {})
    offer_str = "<" + ",".join([str(v) for v in offer.values()]) + ">" if offer else "<>"


    u = node.get("u", None)
    r = node.get(f"expected_reward_{main_viewer}", None)
    f = node.get(f"future_value_{main_viewer}", None)
    times = len(node.get("total_value", "?"))
    # times = node.get("times", "?")

    connector = "└── " if is_last else "├── "


    u_str = f", u={u:.3f}" if u is not None else ""
    r_str = f", r={r:.3f}" if r is not None else ""
    f_str = f", future={f:.3f}" if f is not None else ""

    print(f"{prefix}{connector}step {depth} ({role}) ALP:{alp:.3f} {offer_str}{u_str}{r_str}{f_str}, ={times}")

    child_prefix = prefix + ("    " if is_last else "│   ")

    children = node.get("children", [])
    for i, child in enumerate(children):
        is_last_child = (i == len(children) - 1)
        print_tree_structure(
            child,
            main_viewer=main_viewer,
            prefix=child_prefix,
            is_last=is_last_child,
            depth=depth + 1,
            max_depth=max_depth,
        )



def distill_data_sample(env_file:str, ufun_file:str, json_file: str, main_viewer: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(env_file, "r", encoding="utf-8") as f:
        scenario = json.load(f)

    ufuns, reward_normalizer, offer_normalizer = load_ufuns(ufun_file)



    

    # Calculate the expected reward starting from the root node.
    root_reward = propagate_rewards_downward(
        node=data,
        ufuns=ufuns,
        main_viewer=main_viewer,
        reward_normalizer=reward_normalizer,
        offer_normalizer=offer_normalizer,
    )
    future_reward = backpropagate_future_expectation(data, main_viewer, gamma=0.9)


    
    for k in scenario['system_prompt']:
        scenario['system_prompt'][k]=''.join(scenario['system_prompt'][k])
        scenario['offer_generation_prompt'][k]=''.join(scenario['offer_generation_prompt'][k])
    scenario['tu_prompt']=''.join(scenario['tu_prompt'])

    # system prompts    
    system_prompts = scenario['system_prompt']

    # t u prompt
    tu_prompt = scenario['tu_prompt']

    # offer prompt
    offer_gen_prompts = scenario['offer_generation_prompt']

    offer_decision_prompts = [
        "\n".join(scenario['offer_decision_prompt']), "\n".join(scenario['offer_decision_prompt_2'])
    ]




    from action_level_recog import convert_tree_to_llamafactory
    return convert_tree_to_llamafactory(system_prompts,
                                 tu_prompt,
                                 offer_gen_prompts,
                                 offer_decision_prompts,
                                 data, main_viewer, 'data.json')


    

def safe_softmax(x, tau=1.0):
    # x: list or 1d-array
    x = np.array(x, dtype=np.float64)
    if tau <= 0:
        tau = 1e-6
    x = x / float(tau)
    x_max = x.max()
    ex = np.exp(x - x_max)  # numeric stable
    s = ex.sum()
    if s == 0:
        ex = np.ones_like(ex)
        s = ex.sum()
    return (ex / s).tolist()

def child_probs_by_value(children, value_key, tau=1.0):
    vals = [float(child.get(value_key, 0.0)) for child in children]
    return safe_softmax(vals, tau=tau)

def backpropagate_future_expectation(
    node: Dict[str, Any],
    main_viewer: str,
    gamma: float = 0.9,
    better_threshold: float = 0.05  
) -> float:
    """
    Calculate the future_value upwards from the leaf node (using u_immediate, which includes ALP).
    At the same time, check at the main_viewer node whether the previous node's offer should be accepted:
    """

    alp = float(node.get("ALP", 0.0) or 0.0)
    u_immediate = float(node.get("u_immediate", 0.0))
    children = node.get("children", [])

    # The original utility of the current node, used for comparison with the future.

    current_raw = float(node.get("u_raw", 0.0))

    # -------------------------------------------------------
    # 1) if leaf
    # -------------------------------------------------------
    if not children:
        node[f"future_value_{main_viewer}"] = u_immediate
        return u_immediate

    # -------------------------------------------------------
    # 2) Recursively calculate the future_value of child nodes (bottom-up).
    # -------------------------------------------------------
    child_probs = child_probs_by_value(
        children,
        value_key=f"future_value_{main_viewer}",
        tau=1.0
    )

    child_vals = []
    for child in children:
        v = backpropagate_future_expectation(child, main_viewer, gamma)
        child_vals.append(v)

    expected_child = sum(p * v for p, v in zip(child_probs, child_vals))
    V = u_immediate + gamma * (1 - alp) * expected_child
    node[f"future_value_{main_viewer}"] = float(V)


    if node.get("role") == main_viewer:
        


        opponent_future_raw = []

        def collect_opponent_raw(n):
            if n.get("role") != main_viewer:
                if "u_raw" in n:
                    opponent_future_raw.append(float(n["u_raw"]))
            for c in n.get("children", []):
                collect_opponent_raw(c)

        collect_opponent_raw(node)

    return V


def propagate_rewards_downward(
    node: Dict[str, Any],
    ufuns: Dict[str, callable],
    main_viewer: str,
    reward_normalizer,
    offer_normalizer,
    accumulated_reward: float = 0.0,
    prob_till_here: float = 1.0
) -> float:
    role = node.get("role")
    offer = node.get("offer", None)
    alp = float(node.get("ALP", 0.0) or 0.0)
    children = node.get("children", [])

    # 1) raw reward
    u_raw = 0.0
    if offer:
        try:
            u_raw = float(reward_normalizer(ufuns[main_viewer](offer)[0]))
        except:
            u_raw = float(reward_normalizer(offer))

    # 2) calculate immediate reward
    u_immediate = alp * u_raw

    # 3) cumulative reward
    cumulative_reward = accumulated_reward + prob_till_here * u_immediate


    node["u_raw"] = u_raw
    node["u_immediate"] = u_immediate
    node[f"expected_reward_{main_viewer}"] = float(cumulative_reward)


    for child in children:
        propagate_rewards_downward(
            child,
            ufuns,
            main_viewer,
            reward_normalizer,
            offer_normalizer,
            accumulated_reward=cumulative_reward,
            prob_till_here=prob_till_here * (1 - alp)
        )

    return cumulative_reward



def load_ufuns(UFUN_FILE: str):
    """
    Dynamically load the utility function modules (e.g., uf_a, uf_b, reward_normalize).
    """

    module_name = os.path.splitext(os.path.basename(UFUN_FILE))[0]
    spec = importlib.util.spec_from_file_location(module_name, UFUN_FILE)
    module_ufun = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_ufun)


    uf_a = getattr(module_ufun, "uf_a", None)
    uf_b = getattr(module_ufun, "uf_b", None)
    reward_normalizer = getattr(module_ufun, "reward_normalize", None)
    offer_normalizer = getattr(module_ufun, "offer_normalize", None)



    return {"a": uf_a, "b": uf_b}, reward_normalizer, offer_normalizer





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare finetune data from negotiation trees")
    parser.add_argument('--data_dir', type=str, required=True, help='path to negotiation tree directory')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')



    args = parser.parse_args()



    DATA_DIR = pathlib.Path(args.data_dir)

    OUTPUT_PATH = pathlib.Path(args.output_dir)

    os.makedirs(OUTPUT_PATH, exist_ok=True)




    
    all_env_stats = {}   


    for subenv in os.listdir(DATA_DIR):


        subenv_dir = DATA_DIR / subenv

        this_env_data = {'a':[],'b':[]}


        for sample in tqdm(os.listdir(subenv_dir),):
            '''
            structure
            |-- env_antique_full@b@1.json
            `-- scenario
                |-- __pycache__
                |   `-- ...................pyc
                |-- env_antique_full.json
                |-- env_antique_full_uf.py
                `-- pareto.json
            '''
            sample_dir = subenv_dir / sample

            d = glob.glob(f"{sample_dir}/*@*@*.json", recursive=True)
            if len(d)==0:
                continue
            dpath = d[0].split('/')[-1]

            
            env, mainviewer = dpath.split('@')[0], dpath.split('@')[1]
            assert env == subenv

            ufun_file = sample_dir / f"scenario/{env}_uf.py"
            assert os.path.exists(ufun_file)

            env_file = sample_dir / f"scenario/{env}.json"
            assert os.path.exists(env_file)

            this_path = d[0]

            final_score = watch_data(this_path)




            this_sample_data = distill_data_sample(env_file, ufun_file, this_path, mainviewer)
            this_env_data[mainviewer].extend(this_sample_data)

        for mainviewer in this_env_data:
            filename = OUTPUT_PATH / f'{subenv}@{mainviewer}.json'
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(this_env_data[mainviewer], f, ensure_ascii=False, indent=2)
            print(f"[DONE] {len(this_env_data[mainviewer])} samples saved → {filename}")


    FINAL_PATH = os.path.join(OUTPUT_PATH, 'final')

    os.makedirs(FINAL_PATH, exist_ok=True)

    all_data = []

    envlist = {
        'env_travel': [],
        'env_citymanagement': [],
        'env_antique_full': []
    }

    for filename in os.listdir(OUTPUT_PATH):
        if filename.endswith('.json'):
            thisenv = filename.split('@')[0]
            assert thisenv in envlist

            file_path = os.path.join(OUTPUT_PATH, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                all_data.extend(data)
                envlist[thisenv].extend(data)

    data_total_path = os.path.join(FINAL_PATH, 'data_total.json')
    with open(data_total_path, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)



    for env in envlist:
        env_data_path = os.path.join(FINAL_PATH, f'data_without_{env}.json')
        thisdata = []
        for k,v in envlist.items():
            if k != env:
                thisdata.extend(v)
        with open(env_data_path, 'w', encoding='utf-8') as f:
            json.dump(thisdata, f, ensure_ascii=False, indent=4)
