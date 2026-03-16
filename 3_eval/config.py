import os
from openai import OpenAI


class CONFIG:
    
    ORIGINAL_LLAMA_NAME = "..."
    
    ORIGINAL_LLAMA_CLIENT = OpenAI(
        api_key="...",
        base_url="..."
    )

    MIIND_NAME = "..."
    
    MIIND_CLIENT = OpenAI(
        api_key="...",
        base_url="..."
    )

    CHATGPT_CLIENT = OpenAI(
        api_key="...",
        base_url="..."
    )

    CHATGPT_MODEL_NAME = 'gpt-5.2'

    DEEPSEEK_CLIENT = OpenAI(
        api_key="...",
        base_url="..."
    )

    DEEPSEEK_MODEL_NAME = 'deepseek-chat'