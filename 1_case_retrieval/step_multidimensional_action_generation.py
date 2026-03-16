from sympy import sequence
from short_term_strategy import ShortTermNode
from load_scenario import load_scenario
import json
import uuid
from typing import Dict, List, Optional
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import jensenshannon
import random
from copy import deepcopy

import importlib.util
from config import CONFIG
import math
from collections import defaultdict

from call_llm import call_llm_jsonformat,call_llm_show_probs,call_llm_many_times




from config import CONFIG
    



PROMPT_SINGLE_DIMENSION = """
You are a negotiation strategy classifier.

Task:
- Given an agent's **{target}**, evaluate it on **one dimension**: {dimension}.
- Choose **exactly one label** from the following options: [{labels}].

Rules:
- Output must be one of the provided labels ONLY.
- Do NOT include explanations or reasoning.

Input:
{text}
"""



PROMPT_CONSISTENCY = """
You are a negotiation consistency evaluator.

Task:
- Determine how consistent the agent's private **thought** and outward **utterance** are.

Definition of consistency:
- "consistent": The utterance fully reflects the thought, no major difference.
- "partial": The utterance partially reflects the thought, some differences but general alignment remains.
- "inconsistent": The utterance contradicts the thought or misrepresents it significantly.

Rules:
- Output must be exactly one of: consistent, partial, inconsistent.
- Do NOT include explanations or reasoning.

Input:
Thought:
{thought}

Utterance:
{utterance}
"""




def evaluate_output(jobj:Dict) -> Dict:
    # Ask LLM individually for each dimension, outputting only one label at a time.

    result = {
        "thought": {},
        "utterance": {},
        "consistency": "unknown"
    }
    thought,utterance=jobj["thought"], jobj["utterance"]

    # Iterate through each dimension and evaluate both thought and utterance separately.
    for dimension, labels in modelling_dict["dimensions"].items():
        for target, text in [("thought", thought), ("utterance", utterance)]:

            prompt = PROMPT_SINGLE_DIMENSION.format(
                target=target,
                dimension=dimension,
                labels=", ".join(f'"{label}"' for label in labels),
                text=text,
                example_label=labels[0]  
            )

            result[target][dimension] = call_llm_show_probs(prompt, labels)



    # Evaluate the consistency between thought and utterance.
    consistency_prompt = PROMPT_CONSISTENCY.format(
        thought=thought,
        utterance=utterance
    )
    result["consistency"] = call_llm_show_probs(consistency_prompt, modelling_dict["consistency"])



    return result    
    


def expected_value(prob_dict, dim):
    val_map = dimension_value_map[dim]
    E = 0.0
    for label, value in prob_dict.items():

        if label == "best_label":
            continue

        label_lower = label.lower()
        if label_lower in val_map:
            prob = value.get("prob", 0.0)
            E += val_map[label_lower] * prob

    return E

def get_emotion_vector(prob_dict):
    """
    Convert the emotion probability dictionary into a vector, ensuring the order matches 
    modelling_dict['dimensions']['emotion'].
    """

    emotion_labels = modelling_dict["dimensions"]["emotion"]
    return np.array([prob_dict[label]["prob"] for label in emotion_labels])

def emotion_js_distance(prob_dict1, prob_dict2):
    """
    Calculate the Jensen-Shannon distance between two emotion probability distributions, naturally normalized to [0, 1].
    """

    v1 = get_emotion_vector(prob_dict1)
    v2 = get_emotion_vector(prob_dict2)

    v1 = np.clip(v1, 1e-12, 1.0)
    v2 = np.clip(v2, 1e-12, 1.0)

    return jensenshannon(v1, v2)

def compute_distance(eval1, eval2):
    # Calculate the distance between two actions.


    dist_thought = 0.0
    dist_utterance = 0.0

    # Thought
    for dim in dimension_value_map.keys():
        if dim == "emotion":
            js_dist = emotion_js_distance(eval1["thought"][dim], eval2["thought"][dim])
            dist_thought += js_dist ** 2
        else:
            E1 = expected_value(eval1["thought"][dim], dim)
            E2 = expected_value(eval2["thought"][dim], dim)
            dist_thought += (E1 - E2) ** 2
    dist_thought = np.sqrt(dist_thought)

    # Utterance
    for dim in dimension_value_map.keys():
        if dim == "emotion":
            js_dist = emotion_js_distance(eval1["utterance"][dim], eval2["utterance"][dim])
            dist_utterance += js_dist ** 2
        else:
            E1 = expected_value(eval1["utterance"][dim], dim)
            E2 = expected_value(eval2["utterance"][dim], dim)
            dist_utterance += (E1 - E2) ** 2
    dist_utterance = np.sqrt(dist_utterance)

    # Final distance
    distance = (dist_thought + dist_utterance) / 2.0
    return distance

def post_execute(final_results, distance_matrix, prompt, target_count=5, threshold=CONFIG.ACTION_DIVERSITY_THRESHOLD):
    """
    Process the final results:
    1. Remove outputs that are too close in distance.
    2. Randomly explore and add new outputs until the target_count is reached.
    """

    def prune_similar_outputs():
        """
        Remove outputs with high similarity based on the distance matrix.
        """

        n = len(final_results)
        keep_indices = [0]  

        for i in range(1, n):
            too_close = False
            for j in keep_indices:
                if distance_matrix[min(i, j), max(i, j)] < threshold:
                    too_close = True
                    break
            if not too_close:
                keep_indices.append(i)

        return [final_results[idx] for idx in keep_indices]
    
    def random_dimension_combo_probs():
        """
        Randomly generate a set of dimensional probability distributions for thought or utterance.
        """

        combo = {}
        for dim, labels in modelling_dict["dimensions"].items():
            chosen = random.choice(labels)
            combo[dim] = {}
            for label in labels:
                combo[dim][label] = {"prob": 1.0 if label == chosen else 0.0}
            combo[dim]["best_label"] = chosen
        return combo

    def random_thought_utterance_pair():
        """
        Return a set of random dimensional distributions containing both thought and utterance.
        """
        return {
            "evaluation":{
                "thought": random_dimension_combo_probs(),
                "utterance": random_dimension_combo_probs(),
                "consistency": {} 
            }
        }
        
    def generate_output_from_dimensions(combo):
        # thought_dims=[f"{dim}:{combo['evaluation']['thought'][dim]['best_label']}" for dim in combo['evaluation']['thought']]
        thought_dims=[f"{combo['evaluation']['thought'][dim]['best_label']}" for dim in combo['evaluation']['thought']]
        thought_dims=', '.join(thought_dims)
        # utterance_dims=[f"{dim}:{combo['evaluation']['utterance'][dim]['best_label']}" for dim in combo['evaluation']['utterance']]
        utterance_dims=[f"{combo['evaluation']['utterance'][dim]['best_label']}" for dim in combo['evaluation']['utterance']]
        utterance_dims=', '.join(utterance_dims)
        


        base = prompt[-1]['content']
        
        # thought
        thought_prompt=deepcopy(prompt)

        thought_prompt.append({
            "role":"user",
            "content":(
                "Now generate your private reasoning ('thought') **in natural language** according to the following tone.\n"
                f"{thought_dims}\n\n"
                "Output format (strict JSON):\n"
                "{\n  \"thought\": \"...\"\n}"
            )
        })

        thought_output = call_llm_jsonformat(thought_prompt, legal_keys=["thought"])

        # utterance
        utterance_prompt=deepcopy(prompt)

        utterance_prompt.append({
            "role":"user",
            "content":(
                "Now generate your utterance based on the previously generated 'thought' and the following dimensions.\n"
                "Ensure the style matches the dimensions and is consistent with your private reasoning.\n"
                f"Thought:\n{thought_output['thought']}\n"
                f"Dimensions:\n{utterance_dims}\n\n"
                "Output format (strict JSON):\n"
                "{\n  \"utterance\": \"...\"\n}"
            )
        })

        utterance_output = call_llm_jsonformat(utterance_prompt, legal_keys=["utterance"])
        

        consistency_prompt = PROMPT_CONSISTENCY.format(
            thought=thought_output["thought"],
            utterance=utterance_output["utterance"]
        )
        combo['evaluation']["consistency"] = call_llm_show_probs(consistency_prompt, modelling_dict["consistency"])

        # Combine thought and utterance
        final_output = {
            "raw_output":{
                "thought": thought_output["thought"],
                "utterance": utterance_output["utterance"]
            },
            "evaluation":combo['evaluation']

        }


        return final_output
            
    
    # prune
    filtered = prune_similar_outputs()


    # If the quantity is insufficient, randomly explore and add more.

    while len(filtered) < target_count:
        combo = random_thought_utterance_pair()

        if np.mean([compute_distance(i["evaluation"], combo["evaluation"]) for i in filtered]) > threshold:
            pass
        else:
            continue
        
        new_output = generate_output_from_dimensions(combo)


        filtered.append(new_output)

    return filtered
    

def generate_offer(
                    role:str,
                    dialogues_:List,
                    obj:Dict,
                    offer_generation_prompt:str,
                    offer_legal_keys:List[str],

                    ufuns:dict,
                    offer_default_value:dict
                   
                   ) -> Dict:
    """
    Generate an offer based on thought and utterance autoregressively.
    """

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

    for attempt in range(3):

        to_gen_offer_prompt_new=deepcopy(to_gen_offer_prompt)
        to_gen_offer_prompt_new[-1]['content'] = to_gen_offer_prompt_new[-1]['content']+\
            f"\n The offers you previously proposed but later rejected yourself are:{[x['offer'] for x in old_bad_offers]}"
        offer = call_llm_jsonformat(to_gen_offer_prompt_new, legal_keys=offer_legal_keys)


        for k,v in offer.items():
            if v is None:
                offer[k] = offer_default_value[role][k]

        if ufuns is not None:
            reward, explanation = ufuns[role](offer)
            explanation_sentence = f"The reward for you is {reward}.\n The explanation is {explanation}.\n" 
        else:
            reward, explanation = offer, ''
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

        votes = call_llm_many_times(to_ask_ensure_prompt, n=times, temperature=0.7)
        yes_count = sum(1 for v in votes if v.strip().lower() == "y")
        yes_proportion = float(yes_count)/float(times)


        if yes_proportion >= CONFIG.ENSURE_OFFER_THRESHOLD:

            return offer
        else:
            # regenerate offer
            old_bad_offers.append({"offer": offer, "reward": reward})





    # After 3 attempts, if it still doesn't pass, select the first offer.

    if len(old_bad_offers) > 0:

        try:
            max_reward = max([x['reward'] for x in old_bad_offers])
            for obj in old_bad_offers:
                if abs(max_reward - obj['reward']) <= 0.01:
                    return obj["offer"]
            else:
                return old_bad_offers[0]['offer']
        except Exception as e:
            return old_bad_offers[0]['offer']
    else:
        return offer_default_value[role]




def generate_n_variants(
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
                        
                        n: int = 5) -> List[ShortTermNode]:
    """
    prompt: List containing the system and all previous dialogues at the beginning.

    offer_generation_prompt: str, The prompt for generating an offer based on thought and dialogue.

    offer_legal_keys: List[str], Legal keys for the offer.

    Generate n outputs with different styles.
    """

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




    parsed_first = call_llm_jsonformat(gen_tu_prompt, legal_keys=tu_legal_keys)


    evaluation = evaluate_output(parsed_first)
    final_results.append({
        "raw_output": parsed_first,
        "evaluation": evaluation
    })
        



    for i in range(1, n):
        
        # olds
        olds=[x["raw_output"] for x in final_results]
        olds=[f"Previous output {idx+1}: Thought:{x['thought']} \n Utterance:{x['utterance']}\n" for idx,x in enumerate(olds)]
        olds="\n".join(olds)
        
        variant_prompt = deepcopy(gen_tu_prompt)
        variant_prompt[-1]['content'] = (
            variant_prompt[-1]['content'] + "\n\n"
            "Now generate a different response that varies in style and strategy compared to the previous outputs. "
            f"Previous outputs:{olds}"
            "The new output must still strictly follow the JSON format."
        )
        parsed_new = call_llm_jsonformat(variant_prompt, legal_keys=tu_legal_keys)

        



        evaluation = evaluate_output(parsed_new)
        final_results.append({
            "raw_output": parsed_new,
            "evaluation": evaluation
        })

    distance_matrix = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i,j] = compute_distance(final_results[i]["evaluation"], final_results[j]["evaluation"])

    np.set_printoptions(precision=4, suppress=True)

    
    for fr in final_results:
        fr['type']='natural'
    
    if n>1:
        final_results = post_execute(final_results,distance_matrix, prompt,target_count=n)

    
    for fr in final_results:
        if 'type' not in fr:
            fr['type']='manmade'
        
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
            style_prob=output['evaluation'],
            last_offer_accept_prob = None,
        )
        all_nodes.append(this_node)
    return all_nodes



    
    



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
    """
        generate 1 output

    """
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

    parsed_first = call_llm_jsonformat(gen_tu_prompt, legal_keys=tu_legal_keys)


    evaluation = evaluate_output(parsed_first)
    final_results.append({
        "raw_output": parsed_first,
        "evaluation": evaluation
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
            style_prob=output['evaluation'],
            last_offer_accept_prob = None,
        )
        all_nodes.append(this_node)
    return all_nodes[0]