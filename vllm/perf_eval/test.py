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
import csv
from decimal import Decimal
import random

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "random_alora"
LORA_NAME = "random_lora"
invocation_string = "<|start_of_role|>certainty<|end_of_role|>"

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

os.environ["VLLM_USE_V1"] = "1"

# get a tokenizer and figure out the vocabulary size
tokenizer = AutoTokenizer.from_pretrained(BASE_NAME)
vocab_size = tokenizer.vocab_size

###################################################################

def gen_rnd_tokens(shape): 
    # the vocabulary size is a global variable 
    return np.random.randint(0, vocab_size, size=shape).tolist()    

###################################################################

async def send(prompt_tokens, ntokens, use_adapter_name=None):
    if use_adapter_name == ALORA_NAME:
        prefix, suffix, model = [], tokenizer(invocation_string)["input_ids"], use_adapter_name
    elif use_adapter_name == LORA_NAME: # for fairness, add invocation tokens to lora prompt as well
        prefix, suffix, model = [], tokenizer(invocation_string)["input_ids"], use_adapter_name
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

async def get_metrics(stats, histograms):

    # Get current Prometheus metrics
    stat_vals = {}
    hist_vals = {}

    metrics = requests.get("http://localhost:8000/metrics/vllm:num_requests_running").text
    for line in metrics.splitlines():
        for stat in stats:
            if line.startswith(stat):
                split_line = line.split("}")
                stat_name = split_line[0]
                stat_vals[stat_name] = Decimal(split_line[1].strip())
                break
        for hist in histograms:
            if line.startswith(hist):
                split_line = line.split("}")
                hist_name = split_line[0]
                hist_vals[hist_name] = Decimal(split_line[1].strip())
                break

    return stat_vals, hist_vals

###################################################################

def subtract_metrics(stats, histograms, earlier_stats, earlier_histograms):

    final_stat_vals = {}
    final_hist_vals = {}
    for stat in stats:
        final_stat_vals[stat] = stats[stat] - earlier_stats[stat]
    for hist in histograms:
        final_hist_vals[hist] = histograms[hist] - earlier_histograms[hist]
    return final_stat_vals, final_hist_vals

###################################################################

def save_metrics(stats, histograms, adapter_name, file_name):

    f = open("/home/lallison/vllm-eval/vllm/perf_eval/"+file_name,"w")

    for stat in stats:
        f.write(stat+"} "+f"{stats[stat]:.6f}"+"\n")
        f.flush()
    for hist in histograms:
        f.write(hist+"} "+f"{histograms[hist]:.6f}"+"\n")
        f.flush()
    f.close()

###################################################################

async def main():

    prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # # Generate random prompt tokens (run once)
    # for p_len in prompt_lens:
    #     random_prompts = [gen_rnd_tokens(p_len)]
    #     with open(f"prompts/random_prompt_len_{str(p_len)}.txt", 'w') as f:
    #         for prompt in random_prompts:
    #             line = ','.join(map(str, prompt))
    #             f.write(line+'\n')

    stats = ["vllm:kv_cache_usage",
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_hits",
            "vllm:prompt_tokens",
            ]

    histograms = ["vllm:iteration_tokens_total",
                "vllm:time_to_first_token_seconds",
                "vllm:time_per_output_token_seconds",
                "vllm:e2e_request_latency_seconds",
                "vllm:request_queue_time_seconds",
                "vllm:request_inference_time_seconds",
                "vllm:request_prefill_time_seconds",
                "vllm:request_decode_time_seconds",
                ]
    
    random_prompts = []
    current_prompt_len = prompt_lens[8] # max 9
    with open(f'prompts/random_prompt_len_{current_prompt_len}.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            random_prompts.append(prompt_tokens)
    
    batch_size = (351104 // (current_prompt_len + 2 + 256 + 4 + 16)) # batch size chosen to saturate GPU memory
                                                                     # kv cache size in tokens // prompt_len + eot + generation + activation + evaluation
    random.seed(42)
    for i in range(1, batch_size):
        random_prompts.append(random.sample(prompt_tokens, k=current_prompt_len)) # shuffle to avoid accidental cache hits

    print("warm up the inference engine")
    warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    print("done warming up!!")
    # earlier_stat_vals, earlier_hist_vals = await get_metrics(stats, histograms) # record metrics for generation + evaluation calls
    
    ADAPTER_NAME = ALORA_NAME # change this to LORA_NAME and load in lora at server startup to test random lora
    # ADAPTER_NAME = LORA_NAME

    # Call the base model
    base_generation_tokens = await send(random_prompts, ntokens=256, use_adapter_name=BASE_NAME)

    # Call the adapter model
    adapter_prompts = [x + y + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(random_prompts, base_generation_tokens)]

    earlier_stat_vals, earlier_hist_vals = await get_metrics(stats, histograms) # record metrics for evaluation call only
    adapter_generation_tokens = await send(adapter_prompts, ntokens=16, use_adapter_name=ADAPTER_NAME) 

    # Get current Prometheus metrics
    adapter_stat_vals, adapter_hist_vals = await get_metrics(stats, histograms)
    
    # Subtract the metrics from the warmup call
    final_stat_vals, final_hist_vals = subtract_metrics(adapter_stat_vals, adapter_hist_vals, earlier_stat_vals, earlier_hist_vals)
    save_metrics(final_stat_vals, final_hist_vals, ADAPTER_NAME, file_name=f"results/alora_prompt_len_{current_prompt_len}_eval.txt")
    

if __name__ == '__main__':
    asyncio.run(main())