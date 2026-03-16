
import pathlib
from short_term_strategy import ShortTermNode
from load_scenario import  load_scenario
from step_pgmcts_iteration import step_2_0

import json
import uuid, os
from typing import Dict, List, Optional
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import jensenshannon
import random

from get_pareto_data import get_pareto_data


import argparse


# ===============================
# pooled inner process
# ===============================
def main_r(ENV_NAME, main_viewers):

    ROOTDIR =  pathlib.Path(env_path)

    INPUT_DIR = ROOTDIR / ENV_NAME
    INPUT_FILE = INPUT_DIR / f"{ENV_NAME}.json"
    UFUN_FILE = INPUT_DIR / f"{ENV_NAME}_uf.py"
    PARETO_FILE = INPUT_DIR / f"pareto.json"
    OUTPUT_PATH = pathlib.Path(output_path) / f'{ENV_NAME}'

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    global rounds, max_depth, mcts_iter_rounds, num_candidates


    tasks = []
    for i in range(rounds):
        for main_viewer in main_viewers:

            sample_id = len(os.listdir(OUTPUT_PATH))
            THIS_OUTPUT_PATH = OUTPUT_PATH / f"sample_{sample_id}"
            os.makedirs(THIS_OUTPUT_PATH)

            step_2_0(
                INPUT_DIR,
                INPUT_FILE,
                UFUN_FILE,
                PARETO_FILE,
                main_viewer,
                THIS_OUTPUT_PATH,
                ENV_NAME,
                max_depth,
                mcts_iter_rounds,
                num_candidates
            )



def run_one(env, mv):
    print(f"START env={env}")
    main_r(env, mv)
    print(f"DONE env={env}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PG-MCTS of negotiation trees")


    parser.add_argument('--tree_num', type=int, help='how many trees to be generated per setting', default=1)
    parser.add_argument('--max_negotiation_round', type=int, help='max negotiation rounds (default=16). For faster results, set this arg to 6, 8, or 10.', default=16)
    parser.add_argument('--candidate_num', type=int, help='number of candidate actions for main viewer (default=5). For faster results, set this arg to 2 or 3.', default=5)
    parser.add_argument('--pgmcts_iterations', type=int, help='number of PG-MCTS iteration rounds (default=80). For faster results, set this arg to 10-20.', default=80)

    parser.add_argument('--env_path', type=str, help='the path of negotiation environments', required=True)
    parser.add_argument('--output_path', type=str, help='the output path for negotiation trees', required=True)
    parser.add_argument('--llama_dir', type=str, help='the directory of local Llama model', required=True)
    parser.add_argument('--llama_url', type=str, help='the url of local Llama model (e.g. http://localhost:xxxx/v1)', required=True)
    parser.add_argument('--llama_api_key', type=str, help='the api key of local Llama model (e.g. token-abc123)', required=True)


    args = parser.parse_args()

    global env_path, output_path,llama_url,llama_dir,llama_api_key
    env_path = args.env_path
    output_path = args.output_path
    llama_url = args.llama_url
    llama_dir = args.llama_dir
    llama_api_key = args.llama_api_key

    from config import CONFIG
    CONFIG.MODEL_NAME = args.llama_dir  
    CONFIG.CLIENT = OpenAI(  
        api_key=args.llama_api_key,
        base_url=args.llama_url
    )





    ENVS = {
        'env_antique_full': ['a', 'b'],
        'env_travel': ['a', 'b'],
        'env_citymanagement': ['a', 'b'],
    }

    global rounds, max_depth, mcts_iter_rounds, num_candidates

    rounds = args.tree_num
    max_depth = args.max_negotiation_round
    mcts_iter_rounds = args.pgmcts_iterations
    num_candidates = args.candidate_num


    for env, mv in ENVS.items():
        run_one(env, mv)

    print("All processes finished.")
