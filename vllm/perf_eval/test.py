import requests
# from openai import OpenAI
from openai import AsyncOpenAI
import time
import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
import numpy as np
import asyncio
import csv
from decimal import Decimal
import random
import math

BASE_NAME = "/nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"

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
# tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.2-8b-instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Large-Instruct-2407")
vocab_size = tokenizer.vocab_size

###################################################################

def gen_rnd_tokens(shape): 
    # the vocabulary size is a global variable 
    return np.random.randint(0, vocab_size, size=shape).tolist()    

###################################################################

async def send(prompt_tokens, ntokens, use_adapter_name=None, manually_time=False):
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

    if ntokens > 0:
        if manually_time:
            start = time.perf_counter()
        completion = await client.completions.create(
            model = model,
            prompt = full_prompt_tokens,
            max_tokens = ntokens,
            extra_body = {
                "min_tokens" : ntokens
            })
        if manually_time:
            end = time.perf_counter()
            manually_measured_latency = end - start
            return [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(completion.choices[i].text)) for i in range(len(completion.choices))], manually_measured_latency
    else:
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

def save_metrics(stats, histograms, adapter_name, file_name, manually_timed_eval_latencies=None):

    f = open("/home/lallison/vllm-eval/vllm/perf_eval/"+file_name,"w")

    for stat in stats:
        f.write(stat+"} "+f"{stats[stat]:.6f}"+"\n")
        f.flush()
    for hist in histograms:
        f.write(hist+"} "+f"{histograms[hist]:.6f}"+"\n")
        f.flush()
    if manually_timed_eval_latencies is not None:
        f.write("manually_timed_eval_latency_sum "+f"{sum(manually_timed_eval_latencies):.6f}"+"\n")
        f.write("manually_timed_eval_latency_avg "+f"{(sum(manually_timed_eval_latencies) / len(manually_timed_eval_latencies)):.6f}"+"\n")
    f.close()

###################################################################

async def main():

    # # Downloading models to hf_cache (run once)
    # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # # model_name = "mistralai/Mistral-Large-Instruct-2407"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    # # Get invocation tokens to update adapter_config.json for new model
    # print(tokenizer(invocation_string)["input_ids"])

    prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
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
    current_prompt_len = prompt_lens[7] # max 9
    # current_prompt_len = 256
    # current_gen_len = gen_lens[7] # max 7
    current_gen_len = 256
    with open(f'prompts/random_prompt_len_{current_prompt_len}.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            random_prompts.append(prompt_tokens)
    
    # kv_cache_size = 351104 
    kv_cache_size = 407984 # per gpu
    num_gpu = 4
    cache_percentage = 0.70
    num_activation_tokens = len(tokenizer(invocation_string)["input_ids"])
    num_eot_tokens = len(tokenizer("<|end_of_text|>\n")["input_ids"])
    num_eval_tokens = 16
    print(f"num_act_tokens: {num_activation_tokens}, num_eot_tokens: {num_eot_tokens}")
    batch_size = math.floor(kv_cache_size * num_gpu * cache_percentage) // (prompt_lens[9] + current_gen_len + num_eot_tokens + num_activation_tokens + num_eval_tokens) 
                                                                 # batch size chosen to saturate GPU memory
                                                                 # fix batch size based on longest length
    random.seed(42)
    for i in range(1, batch_size):
        random_prompts.append(random.sample(prompt_tokens, k=current_prompt_len)) # shuffle to avoid accidental cache hits

    print("warm up the inference engine")
    warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    print("done warming up!!")
    
    ADAPTER_NAME = ALORA_NAME # change this to LORA_NAME and load in lora at server startup to test random lora
    # ADAPTER_NAME = LORA_NAME

    # Call the base model
    base_start_stat_vals, base_start_hist_vals = await get_metrics(stats, histograms)
    base_generation_tokens = await send(random_prompts, ntokens=current_gen_len, use_adapter_name=BASE_NAME)

    # Call the adapter model
    adapter_prompts = [x + y + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(random_prompts, base_generation_tokens)]

    adapter_start_stat_vals, adapter_start_hist_vals = await get_metrics(stats, histograms)
    adapter_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME) 

    # # Call the base model again
    # base_2_prompts = [x + y + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(adapter_prompts, adapter_generation_tokens)]

    base_2_start_stat_vals, base_2_start_hist_vals = await get_metrics(stats, histograms)
    # base_2_generation_tokens = await send(base_2_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME) 

    # base_2_end_stat_vals, base_2_end_hist_vals = await get_metrics(stats, histograms)
    
    # Subtract the metrics from the warmup call
    # base_1_final_stat_vals, base_1_final_hist_vals = subtract_metrics(adapter_start_stat_vals, adapter_start_hist_vals, base_start_stat_vals, base_start_hist_vals)
    # save_metrics(base_1_final_stat_vals, base_1_final_hist_vals, ADAPTER_NAME, file_name=f"results/lora_gen_len_{current_gen_len}_gen_1.txt")
    adaptor_final_stat_vals, adaptor_final_hist_vals = subtract_metrics(base_2_start_stat_vals, base_2_start_hist_vals, adapter_start_stat_vals, adapter_start_hist_vals)
    save_metrics(adaptor_final_stat_vals, adaptor_final_hist_vals, ADAPTER_NAME, file_name=f"results/alora_prompt_len_{current_prompt_len}_eval.txt")
    # base_2_final_stat_vals, base_2_final_hist_vals = subtract_metrics(base_2_end_stat_vals, base_2_end_hist_vals, base_2_start_stat_vals, base_2_start_hist_vals)
    # save_metrics(base_2_final_stat_vals, base_2_final_hist_vals, ADAPTER_NAME, file_name=f"results/lora_gen_len_{current_gen_len}_gen_2.txt")
    
###################################################################

async def main_poisson():

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
    current_prompt_len = 256
    current_gen_len = 256
    with open(f'prompts/random_prompt_len_{current_prompt_len}.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            random_prompts.append(prompt_tokens)
    
    lambdas = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000] # requests per second
    LAMBDA = lambdas[8] # max 8
    TOTAL_REQUESTS = 300 # reasonably large value
    random.seed(42)
    for i in range(1, TOTAL_REQUESTS):
        random_prompts.append(random.sample(prompt_tokens, k=current_prompt_len)) # shuffle to avoid accidental cache hits

    # generate inter-arrival times according to poisson dist
    np.random.seed(42)
    inter_arrival_times = np.random.exponential(1 / LAMBDA, size=TOTAL_REQUESTS)
    print(f"max interarrival time: {max(inter_arrival_times)}")

    print("warm up the inference engine")
    warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    print("done warming up!!")
    
    ADAPTER_NAME = ALORA_NAME # change this to LORA_NAME and load in lora at server startup to test random lora
    # ADAPTER_NAME = LORA_NAME

    # Get starting Prometheus metrics. For async, can only use Prometheus to record metrics for gen + eval
    earlier_stat_vals, earlier_hist_vals = await get_metrics(stats, histograms) 

    tasks = []
    eval_latencies = []
    for i, delay in enumerate(inter_arrival_times):
        await asyncio.sleep(delay)

        async def gen_eval_send(prompts, gen_len, eval_len):
            """
            Asynchronous function to call base model with prompt p to get generation g, then call adaptor model with prompt (p + g).

            Returns: Latency of evaluation task (manually timed using time.perf_counter())
            """
            # Call the base model
            base_generation_tokens = await send(prompts, ntokens=gen_len, use_adapter_name=BASE_NAME)

            # Call the adapter model
            adapter_prompts = [x + y + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(prompts, base_generation_tokens)]
            adapter_generation_tokens, eval_latency = await send(adapter_prompts, ntokens=eval_len, use_adapter_name=ADAPTER_NAME, manually_time=True) 

            return eval_latency
        
        task = asyncio.create_task(gen_eval_send(prompts=[random_prompts[i]], gen_len=current_gen_len, eval_len=16))
        tasks.append(task)

    eval_latencies = await asyncio.gather(*tasks) # wait until all requests have finished

    # Get current Prometheus metrics
    adapter_stat_vals, adapter_hist_vals = await get_metrics(stats, histograms)
    
    # Subtract the metrics from the warmup call
    final_stat_vals, final_hist_vals = subtract_metrics(adapter_stat_vals, adapter_hist_vals, earlier_stat_vals, earlier_hist_vals)
    save_metrics(final_stat_vals, final_hist_vals, ADAPTER_NAME, file_name=f"results/alora_async_poisson_{LAMBDA}rps.txt", manually_timed_eval_latencies=eval_latencies)


###################################################################

if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(main_poisson())