from email import message
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional
from short_term_strategy import ShortTermNode
import re
from config import CONFIG

import math
from collections import defaultdict



def make_messages(prompt, system_prompt:Optional[str]=None):
    if isinstance(prompt,List):
        messages=prompt
    else:
        assert isinstance(prompt,str)
        if not (system_prompt is None):
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
    return messages


def fix_json_like_string(s: str):
    # Convert key: value to key: "value"
    # Prerequisite: value is not a number, true/false/null

    pattern = r'(".*?")\s*:\s*([A-Za-z_][A-Za-z0-9_-]*)'
    replacement = r'\1: "\2"'
    return re.sub(pattern, replacement, s)

def call_llm_jsonformat(prompt, temperature:int = 0.2, legal_keys:Optional[list[str]] = None):


    messages = make_messages(prompt, "You are a JSON-only assistant. Output valid JSON only.")

    temperature = temperature

    iffind=False
    retry=False
    while not iffind:
        response = CONFIG.CLIENT.chat.completions.create(
            model=CONFIG.MODEL_NAME,
            messages=messages,
            temperature=(1.0+temperature)/2
        )
        output = response.choices[0].message.content

        try:

            start = output.find("{")
            end = output.rfind("}")
            jobj=None
            
            if start != -1 and end != -1:
                clean_json_str = output[start:end + 1]
                try:
                    jobj = json.loads(clean_json_str)
                except Exception as e:
                    fixed_str = fix_json_like_string(clean_json_str)
                    jobj = json.loads(fixed_str)


            elif end==-1:
                output=output+'}'
                start = output.find("{")
                end = output.rfind("}")
                clean_json_str = output[start:end + 1]
                jobj = json.loads(clean_json_str)

            jobj = {k.strip(): v for k, v in jobj.items()}
            
            if jobj is not None: 
                if not legal_keys:
                    return jobj
                
                if not all([k in jobj for k in legal_keys]):

                    pass
                    
                else:
                    return jobj
                
                
        except json.JSONDecodeError:

            retry = True




def call_llm_show_probs(prompt, list_of_candidates: list,):
    """
    Call LLM and calculate the probability for each candidate label (supports multi-token labels, case-insensitive).
    """

    client = CONFIG.CLIENT

    messages = make_messages(prompt, "You are a helpful assistant.")

    
    response = client.chat.completions.create(
        model=CONFIG.MODEL_NAME,
        messages=messages,
        max_tokens=10,       
        temperature=0.5,
        n=1,
        logprobs=True,
        top_logprobs=5
    )

    tokens_info = response.choices[0].logprobs.content

    # Split the candidate label into a sequence of tokens.
    candidate_token_map = {label: label.split() for label in list_of_candidates}

    # Calculate the probability for each candidate label.
    label_probs = {}  
    for label, tokens in candidate_token_map.items():
        logprob_sum = 0.0
        matched = True

        for i, expected_token in enumerate(tokens):
            if i >= len(tokens_info):
                matched = False
                break

            token_info = tokens_info[i]

            found = False
            for top in token_info.top_logprobs:
                if top.token.strip().lower() == expected_token.lower():
                    logprob_sum += top.logprob
                    found = True
                    break

            if not found:
                matched = False
                break

        if matched:
            label_lower = label.lower()
            label_probs[label_lower] = label_probs.get(label_lower, 0.0) + math.exp(logprob_sum)

    # normalize the probabilities
    total_prob = sum(label_probs.values())
    normalized_probs = {}
    for label in list_of_candidates:
        lbl_lower = label.lower()
        prob = label_probs.get(lbl_lower, 0.0)
        normalized_probs[label] = prob / total_prob if total_prob > 0 else 0.0

    # calculate logprob and prepare result
    result = {}
    for label in list_of_candidates:
        lbl_lower = label.lower()
        logprob_val = math.log(label_probs.get(lbl_lower, 1e-12))
        result[label] = {
            # "logprob": logprob_val,
            "prob": normalized_probs[label]
        }

    # best label
    best_label = max(result, key=lambda x: result[x]["prob"])
    result["best_label"] = best_label


    return result



def call_llm_many_times(prompt, temperature=0.7, n=10):

    client = CONFIG.CLIENT
    messages=make_messages(prompt)
    

    response = client.chat.completions.create(
        model=CONFIG.MODEL_NAME,
        messages=messages,
        max_tokens=1024,       
        temperature=temperature,
        n=n,

    )
    
    ans = [x.message.content for x in response.choices]


    return ans
