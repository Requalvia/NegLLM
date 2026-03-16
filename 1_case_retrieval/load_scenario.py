from email import message
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional
from short_term_strategy import ShortTermNode

from config import CONFIG
from call_llm import call_llm_jsonformat




def load_scenario(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)



def safe_parse_json(raw_output):

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        raise e



