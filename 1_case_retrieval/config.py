import json
import os
from openai import OpenAI
from typing import Dict, List, Optional
from short_term_strategy import ShortTermNode

class CONFIG:
    
    MODEL_NAME = "..."
    
    CLIENT = OpenAI(
        api_key="...",
        base_url="..."
    )


    # single-time acceptance probability threshold to stop
    ACCEPT_PROB_SINGLE = 0.70

    # total acceptance probability threshold to stop
    ACCEPT_PROB_TOTAL = 0.95

    ACTION_DIVERSITY_THRESHOLD = 0.075

    ENSURE_OFFER_THRESHOLD = 0.50
