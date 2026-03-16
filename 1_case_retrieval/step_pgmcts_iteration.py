
from copy import deepcopy

from tqdm import tqdm
from transformers import OneFormerForUniversalSegmentation
from call_llm import call_llm_show_probs
from load_scenario import load_scenario
import json
import uuid
from typing import Dict, List, Optional
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import jensenshannon
import random

from step_multidimensional_action_generation import generate_n_variants, generate_1_variants

from short_term_strategy import ShortTermNode, NegotiationTree

import importlib.util
import os
import math
from step_rollout import rollout, call_call_back
from get_pareto_data import get_pareto_data, pareto_score, clear_pareto_cache

from calc_reach_prob import calc_reach_prob
from customize_scenario import make_scenario




from config import CONFIG



def if_reach_agreement(a,b)->bool:
    pass


def if_finish_offer(
    scenario,
    speaker,
    node,
    ufuns,
    dialogue_history,
):
    # The node should be the opponent's node, and I am at the next step after the node. Check if I accept node.offer.
    # If the opponent made an offer last time, we need to ask whether I accept this offer (Y/N),
    # track the probability of acceptance, and record the acceptance rate.



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
        ans_1=call_llm_show_probs(to_ask_accept_prompt_1, ['y','n'])

        # 2
        to_ask_accept_prompt_2=deepcopy(to_ask_accept_prompt)
        to_ask_accept_prompt_2.append({
            "role":"user",
            "content":offer_decision_prompt_2.replace("{offer}",json.dumps(op_last_offer))\
                .replace("{utility}",str(reward))\
                .replace("{explanation}",explanation)
        })
        ans_2=call_llm_show_probs(to_ask_accept_prompt_2, ['y','n'])


        ans1yprob,ans2yprob=ans_1['y']['prob'],ans_2['y']['prob']
        last_offer_accept_prob=(ans1yprob+ans2yprob)*0.50



        if last_offer_accept_prob < 0.005:
            return 0.0
        return last_offer_accept_prob

                    

def step_2_0(   
                ENV_DIR:str,
                INPUT_FILE:str,
                UFUN_FILE:str,
                PARETO_FILE:str,
                main_viewer:str,step_2_output_path:str,
                env_name:str,
                max_depth:int = 10,
                mcts_iter_rounds:int = 50,
                num_candidates:int = 5,
             ):
    

    INPUT_FILE, UFUN_FILE, PARETO_FILE = make_scenario(
        ENV_DIR,
        INPUT_FILE,
        UFUN_FILE,
        PARETO_FILE,
        step_2_output_path,
    )


    

    get_pareto_data(PARETO_FILE)

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

    offer_default_values = scenario['offer_default_values']



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

    
    # get opener 
    opener=(list(scenario['greetings'].keys()))[0]
    opener_greetings=scenario['greetings'][opener]
    

    # thought & utterance prompt
    tu_prompt=scenario['tu_prompt']
    tu_legal_keys=scenario['tu_legal_keys']

    greeting_offer=json.loads(scenario['greeting_offer'])
    
    

    root_node = ShortTermNode(
        role=opener,
        step=0,
        thought="Initial greetings.",
        utterance=opener_greetings,
        offer=greeting_offer,
        is_terminal=False,
        max_depth=max_depth
    )
    

    offers={x:None for x in roles}


    other_viewer = [x for x in roles if x != main_viewer][0]
    ntree = NegotiationTree(
        root_node, main_viewer
    )




    for step in tqdm(range(1 , mcts_iter_rounds)):



       # If all leaves are either at the maximum depth or terminal, then all can be terminated.

        available_leaf_nodes = [x for x in ntree.get_leaf_nodes() if x.step<=max_depth and x.is_terminal is False]
        if len(available_leaf_nodes) == 0:
            break

        # The main viewer moves first.
        # However, if there is no action node for the main viewer (which can only happen in the first round),
        # then let the opponent move first.

        op_leaf_nodes = [
            x for x in ntree.get_leaf_nodes()
            if x.is_terminal == False and (x.role != main_viewer and x.role != 'unexplored')
        ]

        
        ended_node_count, expanded_node_count=0,0
        for oln in op_leaf_nodes:
            # Generate the preceding dialogue.
            environment_prompt = scenario["system_prompt"][main_viewer]
            dialogue_history=[
                {
                    "role": 'system',
                    "content":(
                        f"{environment_prompt}"
                        # f'Your personal info is {agent_info}'
                        f'The dialogue history is\n {oln.build_dialogue_history(main_viewer)}'
                        f"\n\nThis is the {oln.step + 1} round of total {max_depth} negotiation rounds."
                    ) 
                }
            ]

            last_offer_accept_prob = if_finish_offer(scenario, main_viewer, oln, ufuns, dialogue_history)  
            # If the termination probability is greater than a certain threshold, consider the agreement reached.
            # Otherwise, continue negotiating and retain the last_offer_accept_prob in the node.
            if last_offer_accept_prob > CONFIG.ACCEPT_PROB_SINGLE or\
                  calc_reach_prob(oln)*(1-last_offer_accept_prob)<= 1- CONFIG.ACCEPT_PROB_TOTAL:

                this_node = ShortTermNode(
                    role=main_viewer,
                    step= oln.step + 1,
                    thought="",
                    utterance="Accept offer. Reach agreement.",
                    offer=oln.offer,
                    is_terminal=True,
                    type='manmade',
                    style_prob=None,
                    last_offer_accept_prob=1.00,
                    max_depth=max_depth
                )
                ntree.add_node(this_node, oln)
                ended_node_count+=1
                continue


            # Generate potential candidate actions.
            # Do not generate yet (to save time). Generate only when this node is actually selected.
            candidates = [
                ShortTermNode(role = "unexplored")
            ]
            for node in candidates:
                node.last_offer_accept_prob = last_offer_accept_prob
                ntree.add_node(node, oln)

        



        # Selection & Expansion
        # Balance exploration and exploitation: reward + distance to Pareto + exploration
        def select_node_to_expand(node: ShortTermNode) -> ShortTermNode:
            """
            Starting from the root node, follow the path with the highest UCB (Upper Confidence Bound) all the way down,
            until finding an expandable (non-terminal) leaf node.
            If all the child nodes of the subtree are terminal, backtrack.
            """



            def descend(current: ShortTermNode) -> Optional[ShortTermNode]:
                """Recursively descend to find expandable nodes; backtrack if necessary."""

                # If the only child of the current node has the role 'unexplored', then explore it.
                if len(current.children)==1:
                    if current.children[0].role == 'unexplored':

                        
                        environment_prompt = scenario["system_prompt"][main_viewer]
                        dialogue_history=[
                            {
                                "role": 'system',
                                "content":(
                                    f"{environment_prompt}"
                                    f'The dialogue history is\n {current.build_dialogue_history(main_viewer)}'
                                    f"\n\nThis is the {current.step + 1} round of total {max_depth} negotiation rounds."
                                ) 
                            }
                        ]
                        

                        candidates = generate_n_variants(
                            main_viewer,
                            current.step + 1,
                            dialogue_history,
                            tu_prompt,
                            tu_legal_keys,
                            offer_generation_prompt,
                            offer_legal_keys,
                            ufuns,
                            modelling_dict,
                            dimension_value_map,
                            n=num_candidates,
                            offer_default_value=offer_default_values,
                            )

                        for node in candidates:
                            node.last_offer_accept_prob = current.children[0].last_offer_accept_prob
                            ntree.add_node(node, current)
                        ntree.remove_node(current.children[0])

                        
                        

                # If the current node is terminal, it cannot be expanded.

                if current.is_terminal:
                    return None

                # If it is a leaf node (and not terminal), it can be expanded.

                if current.is_leaf():
                    return current

                # If it is a main_viewer node, can only move to the only child node

                if current.role == main_viewer:
                    assert len(current.children) == 1
                    return descend(current.children[0])

                # Otherwise (for opponent's node), select the child node with the highest UCB from the non-terminal ones.
                non_terminal_children = [c for c in current.children if not c.is_terminal]

                # If all nodes are terminal, this branch cannot be expanded → backtrack.
                if not non_terminal_children:
                    return None

                # UCB order 
                for child in sorted(non_terminal_children, key=calc_weight, reverse=True):
                    res = descend(child)
                    if res is not None:  
                        return res


                return None

            result = descend(node)
            if result is None:

                return node  
            else:

                return result

        def calc_weight(node: ShortTermNode) -> float:

            depth_index = math.atan(math.floor((node.step) / 2.0  +1.0 )) / (math.pi / 2)
            part1 = depth_index * node.get_avg_rewards()
            part2 = (1 - depth_index) * pareto_score(node.offer, offer_normalizer)

            # visit count
            parent_visits = node.parent.parent.get_visit_count() if node.parent.parent else 1
            node_visits = node.get_visit_count()

            
            # The exploration coefficient should gradually decrease with MCTS, smoothly reducing from 1.
            expl = 0.1 + 0.9 * (1 + math.cos(math.pi * step / mcts_iter_rounds)) / 2

            part3 = math.sqrt(2 * math.log(parent_visits) /( node_visits+0.99)  )
            total = part1 + part2 + expl * part3

            return total



        

        selected_node = select_node_to_expand(ntree.root)



        # rollout
        reward = rollout(
            scenario=scenario,
            node=selected_node,
            main_viewer=main_viewer,
            other_viewer=other_viewer,
            ufuns=ufuns,
            tu_prompt=tu_prompt,
            tu_legal_keys=tu_legal_keys,
            offer_generation_prompt=offer_generation_prompt,
            offer_legal_keys=offer_legal_keys,
            depth_limit= max_depth - selected_node.step,
            num_leafs_for_main=num_candidates,
            reward_normalizer=reward_normalizer,
            if_finish_offer=if_finish_offer,
            modelling_dict=modelling_dict,
            dimension_value_map=dimension_value_map,
            offer_default_value=offer_default_values,
        )

        # backpropagation
        call_call_back(reward, selected_node)


        is_terminal = False
        if all([offers[x] is not None for x in roles]) is True:
            for item in offers[main_viewer]:
                if offers[main_viewer][item] == offers[other_viewer][item]:
                    pass
                else:
                    break
            else:
                is_terminal=True
            


        mv_to_expend_node = selected_node

        environment_prompt_ov = scenario["system_prompt"][other_viewer]
        dialogue_history_ov=[
            {
                "role": 'system',
                "content":(
                    f"{environment_prompt_ov}"
                    f'The dialogue history is\n {mv_to_expend_node.build_dialogue_history(other_viewer)}'
                    f"\n\nThis is the {mv_to_expend_node.step + 1} round of total {max_depth} negotiation rounds."
                ) 
            }
        ]

        # If this node does not have last_offer_accept_prob, it needs calculate
        last_offer_accept_prob = if_finish_offer(scenario, other_viewer, mv_to_expend_node, ufuns, \
                                                dialogue_history_ov)
        if last_offer_accept_prob > CONFIG.ACCEPT_PROB_SINGLE or \
            calc_reach_prob(mv_to_expend_node)*(1-last_offer_accept_prob)<= 1 - CONFIG.ACCEPT_PROB_TOTAL:

            this_node = ShortTermNode(
                role=other_viewer,
                step= selected_node.step + 1,
                thought="",
                utterance="Accept offer. Reach agreement.",
                offer=mv_to_expend_node.offer,
                is_terminal=True,
                type='manmade',
                style_prob=None, 
                last_offer_accept_prob=1.00,
                max_depth=max_depth
            )
            ntree.add_node(this_node, mv_to_expend_node)
            continue

        output = generate_1_variants(
            other_viewer,
            selected_node.step + 1,
            dialogue_history_ov,
            tu_prompt,
            tu_legal_keys,
            offer_generation_prompt,
            offer_legal_keys,
            ufuns,
            modelling_dict,
            dimension_value_map,
            offer_default_value=offer_default_values
        )

        if output.offer is not None:
            offers[other_viewer] = output.offer
        else:
            output.offer = offers[other_viewer]

        is_terminal=False
        if all([offers[x] is not None for x in roles]) is True:

            for item in offers[other_viewer]:
                if offers[other_viewer][item] == offers[other_viewer][item]:
                    pass
                else:
                    break
            else:
                is_terminal=True
        output.is_terminal = is_terminal
        output.last_offer_accept_prob = last_offer_accept_prob

        ntree.add_node(output, mv_to_expend_node)




    final_path = os.path.join(step_2_output_path, f"{env_name}@{main_viewer}@{len(os.listdir(step_2_output_path))}.json")
    ntree.save_to_json(final_path)
    clear_pareto_cache()
        
    
