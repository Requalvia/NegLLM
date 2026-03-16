'''
    1v1 test
'''

import pathlib

from tqdm import tqdm
from openai import OpenAI
from test_one import rollout
import json



def main_test(ROOTDIR:str, 
            ENV_NAME:str,
            output_path:str, 
            model_a,
            model_b,
            times:int,
            temp_folder:str,
            output_name:str,
            ):
    
    INPUT_DIR = ROOTDIR / "envs" / ENV_NAME
    INPUT_FILE = INPUT_DIR / f"{ENV_NAME}.json"
    UFUN_FILE = INPUT_DIR / f"{ENV_NAME}_uf.py"
    results = []
    for i in tqdm(range(times)):
        result = rollout(
            INPUT_FILE, 
            UFUN_FILE,
            model_a,
            model_b,
            maxsteps,
            temp_folder,
        )
        results.append(result)
    
    with open(output_path / output_name,'w') as jf:
        jf.write(json.dumps(results, indent = 4))

    

import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def run_one_rollout(i, INPUT_FILE, UFUN_FILE, model_a_path, model_b_path, temp_folder):
    from openai import OpenAI  
    
    model_a = {
        "path": model_a_path['path'],
        "client": OpenAI(api_key="token-abc123", base_url=model_a_path['client'])
    }
    model_b = {
        "path": model_b_path['path'],
        "client": OpenAI(api_key="token-abc123", base_url=model_b_path['client'])
    }

    return rollout(INPUT_FILE, UFUN_FILE, model_a, model_b, 16, temp_folder)



def main_test_multip(
        ROOTDIR: str,
        ENV_NAME: str,
        output_path: str,
        model_a: dict,
        model_b: dict,
        times: int,
        temp_folder: str,
        output_name: str
):
    num_workers = 1

    INPUT_DIR = ROOTDIR / "envs" / ENV_NAME
    INPUT_FILE = INPUT_DIR / f"{ENV_NAME}.json"
    UFUN_FILE = INPUT_DIR / f"{ENV_NAME}_uf.py"



    func = partial(
        run_one_rollout,
        INPUT_FILE=INPUT_FILE,
        UFUN_FILE=UFUN_FILE,
        model_a_path=model_a,
        model_b_path=model_b,
        temp_folder=temp_folder,
    )


    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(func, range(times)), total=times))

    output_path = pathlib.Path(output_path)
    with open(output_path / output_name, 'w') as jf:
        jf.write(json.dumps(results, indent=4))

    return results



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Test between two models")
    parser.add_argument('--env_dir', type=str, required=True, help='path to negotiation environment directory')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--temp_dir', type=str, required=True, help='path to temp directory')
    parser.add_argument('--times', type=int, default=1, help='test times per setting')

    args = parser.parse_args()

    ROOTDIR = pathlib.Path(args.env_dir)
    OUTPUT_PATH = pathlib.Path(args.output_dir)
    temp_folder = pathlib.Path(args.temp_dir)

    
    times = args.times
    global maxsteps
    maxsteps = 16
    
    import os
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    MODEL_A = {
        'path': "path/to/miind",
        'client': "http://localhost:xxxx/v1"
        
    }


    MODEL_B = {
        'path': "path/to/llama",
        'client': "http://localhost:xxxx/v1"
    }

    MODEL_A = MODEL_B



    ENV_NAMES = {
        'env_antique_full': ['a', 'b'],
        'env_travel': ['a', 'b'],
        "env_citymanagement": ['a', 'b']
    }
    


    for ENV_NAME,v in ENV_NAMES.items():

        # original original
        o_vs_o = f'{ENV_NAME}_step{maxsteps}_(a=b=original).json'

        if not os.path.exists(OUTPUT_PATH / o_vs_o):
            main_test_multip(
                ROOTDIR,ENV_NAME,OUTPUT_PATH,
                MODEL_B, MODEL_B,
                times,temp_folder,
                o_vs_o
            )

        

        # ft otiginal
        ft_vs_o = f'{ENV_NAME}_step{maxsteps}_(a=ft,b=original).json'
        if not os.path.exists(OUTPUT_PATH / ft_vs_o):
            main_test_multip(
                ROOTDIR,ENV_NAME,OUTPUT_PATH,
                MODEL_A, MODEL_B,
                times,temp_folder,
                ft_vs_o
            )


        # original ft
        o_vs_ft = f'{ENV_NAME}_step{maxsteps}_(a=original,b=ft).json'
        if not os.path.exists(OUTPUT_PATH / o_vs_ft):
            main_test_multip(
                ROOTDIR,ENV_NAME,OUTPUT_PATH,
                MODEL_B, MODEL_A, 
                times,temp_folder,
                o_vs_ft
            )

        ft_vs_ft = f'{ENV_NAME}_step{maxsteps}_(a=ft,b=ft).json'
        if not os.path.exists(OUTPUT_PATH / ft_vs_ft):
            main_test_multip(
                ROOTDIR,ENV_NAME,OUTPUT_PATH,
                MODEL_A, MODEL_A, 
                times,temp_folder,
                ft_vs_ft
            )


