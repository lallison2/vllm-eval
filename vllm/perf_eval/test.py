import requests
# from openai import OpenAI
from openai import AsyncOpenAI
import time
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import numpy as np
import asyncio

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "new_alora" #"ibm-granite/granite-3.2-8b-alora-uncertainty"
invocation_string = "<|start_of_role|>certainty<|end_of_role|>"

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

os.environ["VLLM_USE_V1"] = "1"

# # download your LoRA adapter to ~/.cache/huggingface/…
# alora_path = snapshot_download(repo_id=ALORA_NAME)

# print(alora_path)

# get a tokenizer and figure out the vocabulary size
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME)
vocab_size = tokenizer.vocab_size

###################################################################

def gen_rnd_tokens(shape): 
    # the vocabulary size is a global variable 
    return np.random.randint(0, vocab_size, size=shape).tolist()    

###################################################################

async def send(prompt_tokens, use_alora, ntokens):
    if use_alora:
        prefix, suffix, model = [], invocation_string, ALORA_NAME
    else:
        prefix = []
        suffix = []
        model = BASE_NAME

    # prepare the list of prompts
    full_prompt_tokens = [ prefix + p + suffix for p in prompt_tokens ]

    # if we are being given an exact number of tokens to produce
    # then the completion request reflects as much
    if ntokens > 0:
        # execute the completions for all of the prompts 
        completion = await client.completions.create(
            model = model,   # note that this is either the base model if no intrinsic is used, or the actual intrinsic to be invoked
            prompt = full_prompt_tokens,
            max_tokens = ntokens,
            extra_body = {
                "min_tokens" : ntokens
            })
    else:
        # otherwise let the model determine when to stop
        completion = await client.completions.create(
            model = model,
            prompt = full_prompt_tokens
            )

    return [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(completion.choices[i].text)) for i in range(len(completion.choices))]

###################################################################

def create_random_adapter(rank, invocation_string=None):
    

###################################################################

async def main():

    print("warm up the inference engine")
    warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    _ = await send(warmup_prompts, use_alora=False, ntokens=250)
    print("done warming up!!")

    # prompts = [
    #     (
    #         "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n"
    #         "<|start_of_role|>assistant<|end_of_role|>"
    #     ),
    # ]
    # # Base model call
    # outputs_base = client.completions.create(model=BASE_NAME,
    #                                         prompt=prompts,  
    #                                         temperature=0, 
    #                                         max_tokens=600)

    # choices = outputs_base.choices
    # generated_text = []
    # for i in range(len(prompts)):
    #     prompt = prompts[i]

    #     generated_text += [outputs_base.choices[i].text]
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text[-1]!r}")

    # prompts_alora = [x + y + "<|end_of_text|>\n"+ invocation_string for x,y in zip(prompts, generated_text)] 

    # # Base model with aLoRA call
    # t0 = time.time()
    # alora_outputs = client.completions.create(model=ALORA_NAME,
    #                                         prompt=prompts_alora, 
    #                                         temperature=0, 
    #                                         max_tokens=10)
    # t = time.time() -t0
    # print(f"Time: {t}")
    # for i in range(len(prompts_alora)):
    #     prompt = prompts_alora[i]
    #     generated_text = alora_outputs.choices[i].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


    # stats = ["vllm:kv_cache_usage",
    #         "vllm:prefix_cache_queries",
    #         "vllm:prefix_cache_hits",
    #         "vllm:prompt_tokens",
    #         ]

    # histograms = ["vllm:iteration_tokens_total",
    #             "vllm:time_to_first_token_seconds",
    #             "vllm:time_per_output_token_seconds",
    #             "vllm:e2e_request_latency_seconds",
    #             "vllm:request_queue_time_seconds",
    #             "vllm:request_inference_time_seconds",
    #             "vllm:request_prefill_time_seconds",
    #             "vllm:request_decode_time_seconds",
    #             ]

    # stat_vals = {}
    # histogram_vals = {}

    # # Get current Prometheus metrics
    # metrics = requests.get("http://localhost:8000/metrics/vllm:num_requests_running").text
    # for line in metrics.splitlines():
    #     for stat in stats:
    #         if line.startswith(stat):
    #             stat_vals[stat] = line.split("}")[-1].strip()
    #             break

    # for stat in stats:
    #     print(stat, stat_vals[stat])

if __name__ == '__main__':
    asyncio.run(main())