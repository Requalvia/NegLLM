
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from short_term_strategy import ShortTermNode, NegotiationTree

import json
import uuid
from typing import Dict, List, Optional
from openai import OpenAI
import numpy as np

import random

import math, random, copy
from typing import List, Dict, Tuple, Any, Optional


from config import CONFIG

def load_scenario(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


from call_any_llm import call_llm_jsonformat, call_llm_many_times, call_llm_show_probs

import os, importlib
from copy import deepcopy
import shutil
from pathlib import Path

def calc_reach_prob(node):

    prob = 1.0
    current = node
    
    while current.parent is not None:
        p_accept = getattr(current, "last_offer_accept_prob", None)
        if p_accept is None:
            raise ValueError(f"Node {current} has no attribute 'last_offer_accept_prob'")
        prob *= (1.0 - p_accept)
        current = current.parent
        
    return prob


def if_finish_offer(
    scenario,
    speaker,
    node,
    ufuns,
    dialogue_history,
):


    offer_decision_prompt=scenario['offer_decision_prompt']
    offer_decision_prompt_2=scenario['offer_decision_prompt_2']

    last_offer_accept_prob = None
    if node.offer is not None:
        op_last_offer = node.offer
        if have_uf_and_explanation:
            reward, explanation = ufuns[speaker](op_last_offer)
        else:
            assert ufuns is None
            reward = op_last_offer
            explanation = ''

        to_ask_accept_prompt=deepcopy(dialogue_history)


        # 1
        to_ask_accept_prompt_1=deepcopy(to_ask_accept_prompt)
        to_ask_accept_prompt_1.append({
            "role":"user",
            "content":offer_decision_prompt.replace("{offer}",json.dumps(op_last_offer))\
                .replace("{utility}",str(reward))\
                .replace("{explanation}",explanation)
        })
        ans_1=call_llm_show_probs(test_models[speaker]['path'], test_models[speaker]['client'], to_ask_accept_prompt_1, ['y','n'])


        # 2
        to_ask_accept_prompt_2=deepcopy(to_ask_accept_prompt)
        to_ask_accept_prompt_2.append({
            "role":"user",
            "content":offer_decision_prompt_2.replace("{offer}",json.dumps(op_last_offer))\
                .replace("{utility}",str(reward))\
                .replace("{explanation}",explanation)
        })
        ans_2=call_llm_show_probs(test_models[speaker]['path'], test_models[speaker]['client'], to_ask_accept_prompt_2, ['y','n'])


        ans1yprob,ans2yprob=ans_1['y']['prob'],ans_2['y']['prob']
        last_offer_accept_prob=(ans1yprob+ans2yprob)*0.50


        if last_offer_accept_prob < 0.005:
            return 0.0
        return last_offer_accept_prob


def call_call_back(reward:float, selected_node:ShortTermNode):
    current_node = selected_node
    while current_node.parent is not None:
        current_node.total_value.append(reward)
        current_node = current_node.parent
    current_node.total_value.append(reward)


def cal_reward_traverse(
    node: ShortTermNode,
    main_viewer: str,
    other_viewer: str,
    ufuns: Optional[dict],
    normalizer,
    offer_normalizer,
    is_terminal,
    pareto_file_path:str,
):  
    with open(pareto_file_path) as jf:
        pareto_dict = json.load(jf)

    def get_pareto_score(offer)->float:
        try:

            pareto_data = pareto_dict
            n_offer = offer_normalizer(offer)
            for o in pareto_data:
                if all([n_offer[k]==o['offer'][k] for k in n_offer]):
                    return o['pareto_score']
            return 0.0
        except:
            return 0.0
    


    rewards = []  
    probs = []    

    pareto_scores = []


    current_node = node
    final_offer = current_node.offer

    if ufuns is not None:
        final_offer_reward = 0.0 if final_offer is None else normalizer(ufuns[main_viewer](final_offer)[0])
    else:
        final_offer_reward = 0.0 if final_offer is None else normalizer(final_offer)
    final_offer_pareto = 0.0 if final_offer is None else get_pareto_score(final_offer)


    final_paths = node.find_paths_to_root()

    for idx, n in enumerate(final_paths[1:]):
        p = n.last_offer_accept_prob
        offer = n.parent.offer
        if ufuns is not None:
            reward = normalizer(ufuns[main_viewer](offer))
        else:
            reward = normalizer(offer)
        probs.append(p)
        rewards.append(reward)
        pareto_scores.append(get_pareto_score(offer))

    exp_rewards = []
    exp_rounds = []   
    exp_pareto = []

    possibility_here = 1.0

    for i, (p, r, ps) in enumerate(zip(probs, rewards, pareto_scores), start=1):
        # reward
        exp_rewards.append(possibility_here * p * r)
        # round number
        exp_rounds.append(possibility_here * p * i)
        # pareto score
        exp_pareto.append(possibility_here * p * ps)

        possibility_here *= (1 - p)


    final_exp_reward = sum(exp_rewards) + possibility_here * final_offer_reward
    final_exp_round  = sum(exp_rounds)  + possibility_here * (len(probs) + 1)
    final_exp_pareto = sum(exp_pareto)  + possibility_here * final_offer_pareto



    return final_exp_reward, final_exp_round, final_exp_pareto




def make_scenario(
        scenario_file:str,
        ufun_file:str,
        output_path:str,
    ):

    env_dir = os.path.dirname(scenario_file)
    assert os.path.exists(env_dir)
    custom_script = Path(env_dir) / 'make_scenario.py'
    index_ = str(uuid.uuid4()) 
    snr_path = Path(output_path) / f'scenario_temp_{index_}'
    os.mkdir(snr_path)


    
    if os.path.exists( custom_script ) is False:
        # 
        pareto_file = Path(env_dir) / 'pareto.json'
        assert os.path.exists(pareto_file)
        return scenario_file, ufun_file, pareto_file

    # 
    spec = importlib.util.spec_from_file_location("custom_make_scenario", custom_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "make"):
        raise AttributeError(f"'make_scenario.py' must define a function named 'make'.")


    pareto_file = None
    if os.path.exists(Path(env_dir) / 'pareto.json'):
        pareto_file = Path(env_dir) / 'pareto.json'
    scenario_file, ufun_file, pareto_file = module.make(
        scenario_file=scenario_file,
        ufun_file=ufun_file,
        output_path=snr_path,
        pareto_file=pareto_file,
        is_test = True,
    )
    return scenario_file, ufun_file, pareto_file



def rollout(INPUT_FILE:str, 
            UFUN_FILE:str,
            model_a,
            model_b,
            MAX_DEPTH: int,
            temp_folder:str,
            ):

    INPUT_FILE, UFUN_FILE, pareto_file = make_scenario(
        INPUT_FILE,
        UFUN_FILE,
        temp_folder
    )
    
    
    global test_models
    test_models = {
        'a': model_a,
        'b': model_b
    }

    scenario = load_scenario(INPUT_FILE)

    for k in scenario['system_prompt']:
        scenario['system_prompt'][k]=''.join(scenario['system_prompt'][k])

    scenario['tu_prompt']=''.join(scenario['tu_prompt'])

    for k in scenario['offer_generation_prompt']:
        scenario['offer_generation_prompt'][k]=''.join(scenario['offer_generation_prompt'][k])


    scenario['offer_decision_prompt']='\n'.join(scenario['offer_decision_prompt'])
    offer_decision_prompt=scenario['offer_decision_prompt']

    scenario['offer_decision_prompt_2']='\n'.join(scenario['offer_decision_prompt_2'])
    offer_decision_prompt_2=scenario['offer_decision_prompt_2']
    
    roles=scenario['roles']


    global have_uf_and_explanation
    have_uf_and_explanation = scenario['have_uf_and_explanation']

    if have_uf_and_explanation:
        # ufuns
        module_ufun = os.path.splitext(os.path.basename(UFUN_FILE))[0]
        spec = importlib.util.spec_from_file_location(module_ufun, UFUN_FILE)
        module_ufun = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_ufun)
        ufuns={k:getattr(module_ufun, scenario['utility_function'][k]) for k in roles}
        reward_normalizer = getattr(module_ufun, 'reward_normalize')
        offer_normalizer = getattr(module_ufun, 'offer_normalize')

        assert reward_normalizer is not None and offer_normalizer is not None
    else:
        ufuns = None

        module_ufun = os.path.splitext(os.path.basename(UFUN_FILE))[0]
        spec = importlib.util.spec_from_file_location(module_ufun, UFUN_FILE)
        module_ufun = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_ufun)

        reward_normalizer = getattr(module_ufun, 'reward_normalize')
        offer_normalizer = None

    
    
    offer_generation_prompt=scenario['offer_generation_prompt']
    offer_legal_keys=scenario['offer_legal_keys']

    modelling_dict = scenario['modelling_dict']
    dimension_value_map = scenario['dimension_value_map']
    offer_default_value = scenario['offer_default_values']

    
    opener=(list(scenario['greetings'].keys()))[0]
    opener_greetings=scenario['greetings'][opener]
    

    tu_prompt=scenario['tu_prompt']
    tu_legal_keys=scenario['tu_legal_keys']
    greeting_offer=json.loads(scenario['greeting_offer'])

    root_node = ShortTermNode(
        role=opener,
        step=0,
        thought="Initial greetings.",
        utterance=opener_greetings,
        offer=greeting_offer,
        is_terminal=False
    )
    

    offers={x:None for x in roles}

    main_viewer = 'a'

    other = [x for x in roles if x != opener][0]
    ntree = NegotiationTree(
        root_node, main_viewer
    )

    return rollout_(
            scenario=scenario,
            node=root_node,
            opener=opener,
            other=other,
            pareto_file_path = pareto_file,

            ufuns=ufuns,
            tu_prompt=tu_prompt,
            tu_legal_keys=tu_legal_keys,
            offer_generation_prompt=offer_generation_prompt,
            offer_legal_keys=offer_legal_keys,
            depth_limit=MAX_DEPTH,
            reward_normalizer=reward_normalizer,
            offer_normalizer=offer_normalizer,

            modelling_dict=modelling_dict,
            dimension_value_map=dimension_value_map,
            offer_default_value = offer_default_value,
        )
    

def rollout_(
    reward_normalizer,
    offer_normalizer,
    scenario: dict,
    opener: str,
    other:str,
    pareto_file_path:str,

    node: ShortTermNode,
    ufuns: Optional[dict],
    tu_prompt: str,
    tu_legal_keys: List[str],
    offer_generation_prompt: List[str],
    offer_legal_keys: List[str],
    modelling_dict,
    dimension_value_map,
    depth_limit,
    offer_default_value,
    
) -> float:



    sim_node = copy.deepcopy(node)
    speaker, other = other, opener  


    is_terminal = False

    p1 = copy.deepcopy(speaker)
    p2 = copy.deepcopy(other)

    offers = {p1: None, p2: None}
    if sim_node.offer is not None:
        offers[speaker] = sim_node.offer

    current_node = sim_node

    for step in range(depth_limit):

        environment_prompt = scenario["system_prompt"][speaker]

        dialogue_history = [
            {
                "role": 'system',
                "content": (
                    f"{environment_prompt}"
                    f'The dialogue history is\n {current_node.build_dialogue_history(speaker)}'
                    f"\n\nThis is the {step} round of total {depth_limit} negotiation rounds."
                )
            }
        ]

        last_offer_accept_prob = 0.0
        if offers[other] is not None:
            last_offer_accept_prob = if_finish_offer(scenario, speaker, current_node, ufuns, dialogue_history)


            if last_offer_accept_prob > CONFIG.ACCEPT_PROB_SINGLE or\
                calc_reach_prob(current_node)*(1-last_offer_accept_prob)<= 1- CONFIG.ACCEPT_PROB_TOTAL:
                this_node = ShortTermNode(
                    role=speaker,
                    step=current_node.step + 1,
                    thought="",
                    utterance="Accept offer. Reach agreement.",
                    offer=offers[other],
                    is_terminal=True,
                    type='manmade',
                    style_prob=None,
                    last_offer_accept_prob=1.0
                )
                current_node.add_child(this_node)
                current_node = this_node
                is_terminal = True
                break


        chosen = generate_1_variants(
            speaker,
            current_node.step + 1,
            dialogue_history,
            tu_prompt,
            tu_legal_keys,
            offer_generation_prompt,
            offer_legal_keys,
            ufuns,
            modelling_dict,
            dimension_value_map,
            offer_default_value

        )
        chosen.last_offer_accept_prob=last_offer_accept_prob

        if chosen.offer is not None:
            offers[speaker] = chosen.offer


        is_terminal = False
        if all([offers[x] is not None for x in offers]):
            match = all(offers[p1][k] == offers[p2][k] for k in offers[p1])
            if match:
                is_terminal = True

        if is_terminal:
            break

        current_node.add_child(chosen)
        current_node = chosen



        speaker, other = other, speaker
    


    
    try:
        final_rewards_p1, exp_rounds_1, pareto_1 = cal_reward_traverse(current_node,p1,p2,ufuns,reward_normalizer,offer_normalizer,is_terminal,pareto_file_path)
    except Exception as e:
        final_rewards_p1 = 0.0
        exp_rounds_1 = depth_limit

    try:
        final_rewards_p2, exp_rounds_2, pareto_2 = cal_reward_traverse(current_node,p2,p1,ufuns,reward_normalizer,offer_normalizer,is_terminal,pareto_file_path)
    except Exception as e:
        final_rewards_p2 = 0.0
        exp_rounds_2 = depth_limit
        






    return {
        p1:final_rewards_p1,
        p2:final_rewards_p2,
        'depth':[exp_rounds_1, exp_rounds_2],
        'pareto':[pareto_1, pareto_2],
    }




def generate_offer(
                    role:str,
                    dialogues_:List,
                    obj:Dict,
                    offer_generation_prompt:str,
                    offer_legal_keys:List[str],

                    ufuns:dict,
                    offer_default_value:dict
                   
                   ) -> Dict:




    thought = obj['raw_output']['thought']
    utterance = obj['raw_output']['utterance']
    
    dialogues=deepcopy(dialogues_)
    
    dialogues.append({
        'role':'assistant',
        'content':f"Thought:{thought}.\n Utterance:{utterance}"
    })
    dialogues.append({
        'role':'user',
        'content':offer_generation_prompt[role]
    })
    
    to_gen_offer_prompt=deepcopy(dialogues)

    old_bad_offers=[] 

    for attempt in range(1):

        to_gen_offer_prompt_new=deepcopy(to_gen_offer_prompt)
        to_gen_offer_prompt_new[-1]['content'] = to_gen_offer_prompt_new[-1]['content']+\
            f"\n The offers you previously proposed but later rejected yourself are:{old_bad_offers}"
        offer = call_llm_jsonformat(test_models[role]['path'], test_models[role]['client'], to_gen_offer_prompt_new, legal_keys=offer_legal_keys)

        # try:
        for k,v in offer.items():
            if v is None:
                offer[k] = offer_default_value[role][k]

        if ufuns is not None:
            reward, explanation = ufuns[role](offer)
            explanation_sentence = f"The reward for you is {reward}.\n The explanation is {explanation}.\n" 
        else:
            explanation_sentence = f""

        
        to_ask_ensure_prompt=deepcopy(to_gen_offer_prompt)
        to_ask_ensure_prompt.extend([
            {'role':'assistant','content':str(offer)},
            {'role':'user','content':
                f"Your offer is {str(offer)}.\n "
                +explanation_sentence
                +"Please confirm whether your offer is what you intend to propose to the other party.\n "
                +"If you think it is correct, output 'Y'; otherwise, output 'N'."
            },
        ])
        times=10

        
        return offer




def generate_1_variants(
                        role: str,
                        step: int,
                        prompt: List[str],
                        tu_prompt:str,
                        tu_legal_keys:List[str],
                        offer_generation_prompt: List[str],
                        offer_legal_keys:List[str],
                        ufuns:Optional[dict],
                        modelling_dict_:dict,
                        dimension_value_map_:dict,
                        offer_default_value:dict,
                        ) -> ShortTermNode:
   
    global modelling_dict, dimension_value_map
    modelling_dict = deepcopy(modelling_dict_)
    dimension_value_map = deepcopy(dimension_value_map_)

    final_results = []

    gen_tu_prompt = deepcopy(prompt)
    gen_tu_prompt.append(
        {
            'role':'user','content':tu_prompt
        }
    )


    parsed_first = call_llm_jsonformat(test_models[role]['path'], test_models[role]['client'], gen_tu_prompt, legal_keys=tu_legal_keys)

    final_results.append({
        "raw_output": parsed_first,
    })


    for fr in final_results:
        fr['type']='natural'
    

    for idx in range(len(final_results)):
        fr=final_results[idx]
        offer = generate_offer(role, prompt, fr, offer_generation_prompt,offer_legal_keys, ufuns, offer_default_value)
        
        fr['offer']=offer
    
    all_nodes=[]
    for output in final_results:
        this_node = ShortTermNode(
            role=role,
            step= step,
            thought=output['raw_output']['thought'],
            utterance=output['raw_output']['utterance'],
            offer=output['offer'],
            is_terminal=False,
            type=output['type'],
            style_prob={},
            last_offer_accept_prob = None,
        )
        all_nodes.append(this_node)
    return all_nodes[0]