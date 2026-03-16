
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional
from short_term_strategy import ShortTermNode
import requests
import re

import math
from collections import defaultdict

from config import CONFIG

import logging
logger = logging.getLogger(__name__)

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

    pattern = r'(".*?")\s*:\s*([A-Za-z_][A-Za-z0-9_-]*)'
    replacement = r'\1: "\2"'
    return re.sub(pattern, replacement, s)



def call_llm_jsonformat(client_path, client, prompt, temperature:int = 0.2, legal_keys:Optional[list[str]] = None):

    messages = make_messages(prompt, "You are a JSON-only assistant. Output valid JSON only.")

    temperature = temperature

    iffind=False
    retry=False
    while not iffind:
        if 'llama' in client_path.lower():
            response = client.chat.completions.create(
                model=client_path,
                messages=messages,
                temperature=(1.0+temperature)/2
            )

        elif 'deepseek' in client_path.lower():

            dsclient = CONFIG.DEEPSEEK_CLIENT
            response = dsclient.chat.completions.create(
                model=CONFIG.DEEPSEEK_MODEL_NAME,
                messages=messages,
                temperature=(1.0+temperature)/2
            )
            
        elif 'chatgpt' in client_path.lower():
            client = CONFIG.CHATGPT_CLIENT
            response = client.responses.create(
                model=CONFIG.CHATGPT_MODEL_NAME,
                input=messages,
                reasoning={"effort": "none"},
            )
        else:
            raise ValueError()

        if 'chatgpt' in client_path.lower():
            output = response.output_text
        else:
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
            
            if jobj is not None: 
                if not legal_keys:

                    return jobj
                
                if not all([k in jobj for k in legal_keys]):
                    pass
                    
                else:
                    return jobj
                
                
        except json.JSONDecodeError:
            print(f"JSONDecodeError:{output}")
            retry = True
        
        print(f"Try again: {output}")



def call_llm_show_probs(client_path, client, prompt, list_of_candidates: list,):
    """
     LLM （ token 、）。

    """



    messages = make_messages(prompt, "You are a helpful assistant.")

    
    # 
    if 'llama' in client_path.lower():
        response = client.chat.completions.create(
            model=client_path,
            messages=messages,
            max_tokens=10,       #  token  token
            temperature=0.2,
            n=1,
            logprobs=True,
            top_logprobs=5
        )
    elif 'deepseek' in client_path.lower():

        dsclient = CONFIG.DEEPSEEK_CLIENT
        response = dsclient.chat.completions.create(
            model=CONFIG.DEEPSEEK_MODEL_NAME,
            messages=messages,
            max_tokens=10,       #  token  token
            temperature=0.2,
            n=1,
            logprobs=True,
            top_logprobs=5
        )
    elif 'chatgpt' in client_path.lower():
        client = CONFIG.CHATGPT_CLIENT

        from collections import Counter

        K = 1
        outputs = []

        for _ in range(K):
            response = client.responses.create(
                model=CONFIG.CHATGPT_MODEL_NAME,       
                input=messages,
                reasoning={"effort": "none"},
            )
            outputs.append(response.output_text.strip().lower())

        counter = Counter(outputs)


        result = {}
        for label in list_of_candidates:
            lbl_lower = label.lower()
            prob = counter.get(lbl_lower, 0) / K
            result[label] = {
                "prob": prob
            }

        best_label = max(result, key=lambda x: result[x]["prob"])
        result["best_label"] = best_label

        return result

    else:
        raise ValueError()

    tokens_info = response.choices[0].logprobs.content


    candidate_token_map = {label: label.split() for label in list_of_candidates}

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


    total_prob = sum(label_probs.values())
    normalized_probs = {}
    for label in list_of_candidates:
        lbl_lower = label.lower()
        prob = label_probs.get(lbl_lower, 0.0)
        normalized_probs[label] = prob / total_prob if total_prob > 0 else 0.0


    result = {}
    for label in list_of_candidates:
        lbl_lower = label.lower()
        logprob_val = math.log(label_probs.get(lbl_lower, 1e-12))
        result[label] = {
            # "logprob": logprob_val,
            "prob": normalized_probs[label]
        }


    best_label = max(result, key=lambda x: result[x]["prob"])
    result["best_label"] = best_label


    return result





def call_llm_many_times(client_path, client, prompt, temperature=0.7, n=10):

    messages=make_messages(prompt)
    

    if 'llama' in client_path.lower():
        response = client.chat.completions.create(
            model=client_path,
            messages=messages,
            max_tokens=1024,   
            temperature=0.2,
            n=n,
        )
        ans = [x.message.content for x in response.choices]
    elif 'deepseek' in client_path.lower():

        dsclient = CONFIG.DEEPSEEK_CLIENT
        ans = []
        for i in range(n):
            response = dsclient.chat.completions.create(
                model=CONFIG.DEEPSEEK_MODEL_NAME,
                messages=messages,
                max_tokens=1024,   
                temperature=0.2,
                n=1,
            )

            ans.append(response.choices[0].message.content  ) 
    elif 'chatgpt' in client_path.lower():
        client = CONFIG.CHATGPT_CLIENT
        ans = []
        for i in range(n):
            response = client.responses.create(
                model=CONFIG.CHATGPT_MODEL_NAME,
                input=messages,
                reasoning={"effort": "none"},
            )
            print(response)
            ans.append(response.output_text ) 

    else:
        raise ValueError()


    
    return ans


