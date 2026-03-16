from short_term_strategy import ShortTermNode
import json
import uuid
from typing import Dict, List, Optional
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import jensenshannon
import random

import math, random, copy
from typing import List, Dict, Tuple, Any, Optional

from step_multidimensional_action_generation import generate_n_variants
from config import CONFIG



from calc_reach_prob import calc_reach_prob


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
    is_terminal
):
    # Calculate the probability of reaching an agreement from the end to the beginning.


    rewards = [] 
    single_rewards = []
    probs = []

    current_node = node
    

    final_offer = current_node.offer
    if ufuns is not None:
        final_offer_reward = 0.0 if final_offer is None else normalizer(ufuns[main_viewer](final_offer)[0])
    else:
        final_offer_reward = 0.0 if final_offer is None else normalizer(final_offer)
    

    final_offer_reach_prob = 1.0




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
    
    exp=[]
    
    possibility_here = 1.0

    for p, r in zip(probs,rewards):

        exp.append(possibility_here*p*r)
        possibility_here*=(1-p)

    final_exp = sum(exp) + possibility_here* final_offer_reward



    return final_exp





# Extract points from the Pareto frontier records.

def pareto_records_to_points(records: List[Dict[str,Any]]) -> List[Tuple[float,float]]:
    return [(r['seller_reward'], r['buyer_reward']) for r in records]



def rollout(
    if_finish_offer,
    reward_normalizer,
    scenario: dict,
    node: ShortTermNode,
    main_viewer: str,
    other_viewer: str,
    ufuns: Optional[dict],
    tu_prompt: str,
    tu_legal_keys: List[str],
    offer_generation_prompt: List[str],
    offer_legal_keys: List[str],
    modelling_dict,
    dimension_value_map,
    offer_default_value,
    depth_limit,
    num_leafs_for_main,
    
    
) -> float:
    """Simulate several steps forward from the current node and return the final reward."""





    sim_node = copy.deepcopy(node)
    speaker, other = other_viewer, main_viewer  
    last_offer = sim_node.offer  

    is_terminal = False

    offers = {main_viewer: None, other_viewer: None}
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
                    # f'Your personal info is {agent_info}'
                    f'The dialogue history is\n {current_node.build_dialogue_history(speaker)}'
                    f"\n\nThis is the {node.step+step} round of total {node.step+depth_limit} negotiation rounds."
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

        # generate candidate actions
        n = num_leafs_for_main if speaker == main_viewer else 1
        outputs = generate_n_variants(
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
            n=1,
            offer_default_value = offer_default_value
            
        )
        chosen = random.choice(outputs)
        chosen.last_offer_accept_prob=last_offer_accept_prob


        if chosen.offer is not None:

            offers[speaker] = chosen.offer


        # check whether reach agreement 
        is_terminal = False
        if all([offers[x] is not None for x in offers]):
            match = all(offers[main_viewer][k] == offers[other_viewer][k] for k in offers[main_viewer])
            if match:
                is_terminal = True

        if is_terminal:
            break

        current_node.add_child(chosen)
        current_node = chosen



        # swap
        speaker, other = other, speaker
    



    
    # backpropagate the final reward
    final_rewards = cal_reward_traverse(current_node,
                                        main_viewer,
                                        other_viewer,
                                        ufuns,
                                        reward_normalizer,
                                        is_terminal)






    return final_rewards

