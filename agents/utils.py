import os

import tiktoken
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from transformers import AutoTokenizer


def load_llm_agent(model_provider, temperature=0.1):
    if model_provider == "azure_openai":
        chat_model = AzureChatOpenAI(deployment_name=os.environ["AZURE_ENGINE"],
                                     openai_api_key=os.environ["AZURE_OPENAI_KEY"],
                                     openai_api_base=os.environ["AZURE_OPENAI_BASE"],
                                     openai_api_version=os.environ["AZURE_OPENAI_VERSION"],
                                     temperature=temperature,
                                     request_timeout=60,
                                     max_retries=10,
                                     openai_api_type="azure")
    elif model_provider == "openai":
        chat_model = ChatOpenAI(temperature=temperature)
    elif model_provider == "llama":
        chat_model = ChatOpenAI(model=os.environ["LLAMA_ENGINE"],
                                openai_api_key=os.environ["LLAMA_API_KEY"],
                                openai_api_base=os.environ["LLAMA_API_BASE"],
                                temperature=temperature,
                                request_timeout=60,
                                max_retries=10)
    else:
        raise NotImplementedError(f"Unsupported LLM provider {model_provider}.")
    return chat_model


def load_tokenizer(model_name):
    if "llama" in model_name:
        if "llama70b" == model_name:
            return AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
        elif "llama13b" == model_name:
            return AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
        else:
            raise NotImplementedError(f"Unsupported llama tokenizer for model {model_name}.")
    else:
        return tiktoken.encoding_for_model(model_name)


def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, model_name="gpt-3.5-turbo") -> (str, bool):
    tokenizer = load_tokenizer(model_name)
    lines = scratchpad.split('\n\n')
    observations = filter(lambda x: x.startswith('Previous Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(tokenizer.encode('\n\n'.join(lines))) > n_tokens and len(observations_by_tokens) > 0:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = '[Truncated Observation]'
    return '\n\n'.join(lines), len(tokenizer.encode('\n\n'.join(lines))) > n_tokens
