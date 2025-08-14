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

async def send(prompt_tokens, ntokens, use_adapter_name=None):
    if use_adapter_name is not None:
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
                stat_vals[stat_name] = split_line[1].strip()
                break
        for hist in histograms:
            if line.startswith(hist):
                split_line = line.split("}")
                hist_name = split_line[0]
                hist_vals[hist_name] = split_line[1].strip()
                break

    return stat_vals, hist_vals

###################################################################

def subtract_warmup_metrics(stats, histograms, warmup_stats, warmup_histograms):

    final_stat_vals = {}
    final_hist_vals = {}
    for stat in stats:
        final_stat_vals[stat] = stats[stat] - warmup_stats[stat]
    for hist in histograms:
        final_hist_vals[hist] = histograms[hist] - warmup_histograms[hist]
    return final_stat_vals, final_hist_vals

###################################################################

def save_metrics(stats, histograms, adapter_name):

    f = open("/home/lallison/vllm-eval/vllm/perf_eval/"+adapter_name+".txt","w")
    for stat in stats:
        f.write(stat+" "+stats[stat]+"\n")
        f.flush()
    for hist in histograms:
        f.write(hist+" "+histograms[hist]+"\n")
        f.flush()
    f.close()

###################################################################

async def main():

    random_prompts = [gen_rnd_tokens(256)]
    with open("random_prompt.txt", 'w') as f:
        for prompt in random_prompts:
            line = ','.join(map(str, prompt))
            f.write(line+'\n')
    
    read_random_prompts = []
    with open('random_prompt.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            read_random_prompts.append(prompt_tokens)
    print(read_random_prompts)

    # print("warm up the inference engine")
    # warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    # _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    # warmup_stat_vals, warmup_hist_vals = await get_metrics(stats, histograms)
    # print("done warming up!!")

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
    
    # ADAPTER_NAME = ALORA_NAME # change this to LORA_NAME to test random lora

    # # Call the base model
    # # _ = await send(warmup_prompts, ntokens=250, use_adapter_name=ADAPTER_NAME)

    # # Call the adapter model

    # # Get current Prometheus metrics
    # alora_stat_vals, alora_hist_vals = await get_metrics(stats, histograms)
    
    # # Subtract the metrics from the warmup call
    # final_stat_vals, final_hist_vals = subtract_warmup_metrics(alora_stat_vals, alora_hist_vals, warmup_stat_vals, warmup_hist_vals)
    # save_metrics(final_stat_vals, final_hist_vals, ADAPTER_NAME)




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
    

if __name__ == '__main__':
    asyncio.run(main())