import aiohttp
import openai
import os
import time
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from typing import Dict, Any
from dotenv import load_dotenv

from .format import Message
from .price import cost_count
from .llm import LLM
from .llm_registry import LLMRegistry


OPENAI_API_KEYS = ['']
BASE_URL = ''

load_dotenv()
# Prefer custom names, fallback to standard OpenAI env names
_RAW_BASE_URL = os.getenv('BASE_URL') or os.getenv('OPENAI_API_BASE')
_RAW_API_KEY = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')


def _build_chat_endpoint(base_url: Optional[str]) -> str:
    """Return a normalized chat completions endpoint.
    - If base_url already contains '/chat/completions', return it.
    - If base_url is like '.../v1', append '/chat/completions'.
    - If base_url is None/empty, use the default OpenAI endpoint.
    """
    default = 'https://api.openai.com/v1/chat/completions'
    if not base_url:
        return default
    base_url = base_url.rstrip('/')
    if base_url.endswith('/chat/completions'):
        return base_url
    # Common cases: 'https://api.openai.com/v1' or provider-compatible base
    return f"{base_url}/chat/completions"


MINE_BASE_URL = _build_chat_endpoint(_RAW_BASE_URL)
MINE_API_KEYS = _RAW_API_KEY


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
async def achat(model_name:str, messages:list):
    request_url = MINE_BASE_URL
    authorization_key = MINE_API_KEYS
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {authorization_key}'
    }
    data = {
        "model": model_name,
        "messages": [m.to_dict() for m in messages],
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(request_url, headers=headers ,json=data) as response:
            # If provider returns HTML (e.g., 404 page), raise a clearer error
            if 'application/json' not in (response.headers.get('Content-Type') or ''):
                text = await response.text()
                raise aiohttp.ContentTypeError(
                    response.request_info,
                    history=response.history,
                    message=f"Unexpected content-type: {response.headers.get('Content-Type')} at {request_url}. Body preview: {text[:200]}"
                )
            response_data = await response.json()
            if 'choices' not in response_data:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"OpenAI API Error: {error_message}")
            prompt = "".join([item.content for item in messages])
            completion = response_data['choices'][0]['message']['content']
            cost_count(prompt, completion, model_name)
            return completion


@LLMRegistry.register('gpt-4o')
@LLMRegistry.register('gpt-4o-mini')
@LLMRegistry.register('gpt-4-turbo')
@LLMRegistry.register("gpt-4-0125-preview")
@LLMRegistry.register("gpt-4-1106-preview")
@LLMRegistry.register("gpt-4-vision-preview")
@LLMRegistry.register("gpt-4-0314")
@LLMRegistry.register("gpt-4-32k")
@LLMRegistry.register("gpt-4-32k-0314")
@LLMRegistry.register("gpt-4-0613")
@LLMRegistry.register("gpt-3.5-turbo-0125")
@LLMRegistry.register("gpt-3.5-turbo-1106")
@LLMRegistry.register("gpt-3.5-turbo-instruct")
@LLMRegistry.register("gpt-3.5-turbo-0301")
@LLMRegistry.register("gpt-3.5-turbo-0613")
@LLMRegistry.register("gpt-3.5-turbo-16k-0613")
@LLMRegistry.register("gpt-3.5-turbo")
@LLMRegistry.register("gpt-4")
class GPTChat(LLM):

    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 1.0, max_tokens: int = 1024):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS
        
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]
        elif isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            messages = [Message(role=m['role'], content=m['content']) for m in messages]
            
        return await achat(self.model_name,messages)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass