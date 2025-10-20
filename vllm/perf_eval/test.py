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

BASE_NAME = "/home/lallison/.cache/huggingface/hub/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"
# BASE_NAME = "/nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
# BASE_NAME = "/nobackup/users/lallison/hf_cache/models--mistralai--Mistral-Large-Instruct-2407/snapshots/a286006d554cb37a61d13c7ae61bc90cc1d372fc"

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

async def send(prompt_tokens, ntokens, use_adapter_name=None, manually_time=False, custom_inv_tokens=None):
    if custom_inv_tokens != None:
        prefix, suffix, model = [], custom_inv_tokens, use_adapter_name
    elif use_adapter_name == ALORA_NAME:
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

        # print(f"sending request with {len(full_prompt_tokens[0])} tokens")
        completion = await client.completions.create(
            model = model,
            prompt = full_prompt_tokens,
            max_tokens = ntokens,
            temperature = 0.0,
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
    # # model_name = "meta-llama/Llama-3.3-70B-Instruct"
    # # model_name = "mistralai/Mistral-Large-Instruct-2407"
    # model_name = "ibm-granite/granite-3.2-8b-instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    # # Generate random prompt tokens (run once)
    # np.random.seed(42)
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

    # current_prompt_len = prompt_lens[3] # max 9
    # current_gen_len = 256

    current_gen_len = gen_lens[0] # max 7
    current_prompt_len = 256

    with open(f'prompts/random_prompt_len_{current_prompt_len}.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            random_prompts.append(prompt_tokens)
    
    kv_cache_size = 351104 # granite 3.2
    # kv_cache_size = 407984 # llama 70b (4 gpu)
    # kv_cache_size = 912688 # mistral large (8 gpu)
    cache_percentage = 1.0
    num_activation_tokens = len(tokenizer(invocation_string)["input_ids"])
    num_eot_tokens = len(tokenizer("<|end_of_text|>\n")["input_ids"])
    num_eval_tokens = 16

    batch_size = math.floor(kv_cache_size * cache_percentage) // (prompt_lens[9] + 256 + num_eot_tokens + num_activation_tokens + num_eval_tokens)
                                                                 # batch size chosen to saturate GPU memory
                                                                 # fix batch size based on longest length
    # batch_size = math.floor(kv_cache_size * cache_percentage) // (current_prompt_len + gen_lens[7] + num_eot_tokens + num_activation_tokens + num_eval_tokens)
    # batch_size = 1

    random.seed(42)
    for i in range(1, batch_size):
        random_prompts.append(random.sample(prompt_tokens, k=current_prompt_len)) # shuffle to avoid accidental cache hits

    np.random.seed(100)
    print("warm up the inference engine")
    warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    print("done warming up!!")
    
    ADAPTER_NAME = ALORA_NAME # change this to LORA_NAME and load in lora at server startup to test random lora
    # ADAPTER_NAME = LORA_NAME

    # ADAPTER_NAME_2 = ALORA_NAME + "_2" # if using multiple adapters
    # ADAPTER_NAME_3 = ALORA_NAME + "_3"
    # ADAPTER_NAME_4 = ALORA_NAME + "_4"
    # ADAPTER_NAME_5 = ALORA_NAME + "_5"

    # ADAPTER_NAME_2 = LORA_NAME + "_2"
    # ADAPTER_NAME_3 = LORA_NAME + "_3"
    # ADAPTER_NAME_4 = LORA_NAME + "_4"
    # ADAPTER_NAME_5 = LORA_NAME + "_5"

    # Call the base model
    base_start_stat_vals, base_start_hist_vals = await get_metrics(stats, histograms)
    base_generation_tokens = await send(random_prompts, ntokens=current_gen_len + num_eot_tokens, use_adapter_name=BASE_NAME) # generate extra tokens to ensure that model doesn't stop early

    # Call the adapter model(s)
    adapter_prompts = [x + y[:current_gen_len] + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(random_prompts, base_generation_tokens)]
    print("random prompt len: ", len(random_prompts[0]))
    print("response len (potentially + eot): ", len(base_generation_tokens[0]))
    print("eot len: ", len(tokenizer("<|end_of_text|>\n")["input_ids"]))

    adapter_start_stat_vals, adapter_start_hist_vals = await get_metrics(stats, histograms)
    adapter_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME)
    # adapter_2_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME_2, custom_inv_tokens=[2, 22, 222, 2222]) 
    # adapter_3_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME_3, custom_inv_tokens=[3, 33, 333, 3333]) 
    # adapter_4_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME_4, custom_inv_tokens=[4, 44, 444, 4444]) 
    # adapter_5_generation_tokens = await send(adapter_prompts, ntokens=num_eval_tokens, use_adapter_name=ADAPTER_NAME_5, custom_inv_tokens=[5, 55, 555, 5555]) 

    # # Call the base model again
    # activation_tokens = tokenizer(invocation_string)["input_ids"]
    # base_2_prompts = [x + activation_tokens + y1 + activation_tokens + y2 + activation_tokens + y3 + activation_tokens + y4 + activation_tokens + y5 + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y1,y2,y3,y4,y5 in zip(adapter_prompts, adapter_generation_tokens, adapter_2_generation_tokens, adapter_3_generation_tokens, adapter_4_generation_tokens, adapter_5_generation_tokens)]

    base_2_start_stat_vals, base_2_start_hist_vals = await get_metrics(stats, histograms)
    # _ = await send(base_2_prompts, ntokens=16, use_adapter_name=BASE_NAME)

    # base_2_end_stat_vals, base_2_end_hist_vals = await get_metrics(stats, histograms)
    
    # # Subtract the metrics from the warmup call
    # base_1_final_stat_vals, base_1_final_hist_vals = subtract_metrics(adapter_start_stat_vals, adapter_start_hist_vals, base_start_stat_vals, base_start_hist_vals)
    # save_metrics(base_1_final_stat_vals, base_1_final_hist_vals, ADAPTER_NAME, file_name=f"results/lora_gen_len_{current_gen_len}_gen_1_granite_5_adapters.txt")


    adaptor_final_stat_vals, adaptor_final_hist_vals = subtract_metrics(base_2_start_stat_vals, base_2_start_hist_vals, adapter_start_stat_vals, adapter_start_hist_vals)
    save_metrics(adaptor_final_stat_vals, adaptor_final_hist_vals, ADAPTER_NAME, file_name=f"results/alora_gen_len_{current_gen_len}_eval_granite.txt")

    # base_2_final_stat_vals, base_2_final_hist_vals = subtract_metrics(base_2_end_stat_vals, base_2_end_hist_vals, base_2_start_stat_vals, base_2_start_hist_vals)
    # save_metrics(base_2_final_stat_vals, base_2_final_hist_vals, ADAPTER_NAME, file_name=f"results/lora_gen_len_{current_gen_len}_gen_2_granite_5_adapters.txt")
    
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
    
    # # Generate random prompt tokens (run once)
    # prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # np.random.seed(12345)
    # for p_len in prompt_lens:
    #     random_prompts = [gen_rnd_tokens(p_len)]
    #     with open(f"prompts/random_prompt_len_{str(p_len)}.txt", 'w') as f:
    #         for prompt in random_prompts:
    #             line = ','.join(map(str, prompt))
    #             f.write(line+'\n')

    random_prompts = []
    current_prompt_len = 512
    current_gen_len = 256
    with open(f'prompts/random_prompt_len_{current_prompt_len}.txt', 'r') as f:
        for line in f:
            prompt_strings = line.strip().split(',')
            prompt_tokens = [int(p) for p in prompt_strings]
            random_prompts.append(prompt_tokens)
    
    lambdas = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000] # requests per second
    LAMBDA = lambdas[2] # max 11
    TOTAL_REQUESTS = 500 # reasonably large value
    random.seed(15)
    np.random.seed(100)
    for i in range(1, TOTAL_REQUESTS):
        random_prompts.append(random.sample(prompt_tokens, k=current_prompt_len)) # shuffle to avoid accidental cache hits

    # generate inter-arrival times according to poisson dist
    random.seed(45)
    np.random.seed(300)
    inter_arrival_times = np.random.exponential(1 / LAMBDA, size=TOTAL_REQUESTS)
    print(f"max interarrival time: {max(inter_arrival_times)}")

    # random.seed(135)
    # np.random.seed(900)
    # print("warm up the inference engine")
    # warmup_prompts = [gen_rnd_tokens(500), gen_rnd_tokens(500)]
    # _ = await send(warmup_prompts, ntokens=250, use_adapter_name=None)
    # print("done warming up!!")
    
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

            num_eot_tokens = len(tokenizer("<|end_of_text|>\n")["input_ids"])

            # Call the base model
            base_generation_tokens = await send(prompts, ntokens=gen_len + num_eot_tokens, use_adapter_name=BASE_NAME)

            # Call the adapter model
            adapter_prompts = [x + y[:gen_len] + tokenizer("<|end_of_text|>\n")["input_ids"] for x,y in zip(prompts, base_generation_tokens)]
            adapter_generation_tokens, eval_latency = await send(adapter_prompts, ntokens=eval_len, use_adapter_name=ADAPTER_NAME, manually_time=True) 

            return eval_latency
        
        task = asyncio.create_task(gen_eval_send(prompts=[random_prompts[i]], gen_len=current_gen_len, eval_len=16))
        tasks.append(task)

    eval_latencies = await asyncio.gather(*tasks) # wait until all requests have finished

    # Get current Prometheus metrics
    adapter_stat_vals, adapter_hist_vals = await get_metrics(stats, histograms)
    
    # Subtract the metrics from the warmup call
    final_stat_vals, final_hist_vals = subtract_metrics(adapter_stat_vals, adapter_hist_vals, earlier_stat_vals, earlier_hist_vals)
    save_metrics(final_stat_vals, final_hist_vals, ADAPTER_NAME, file_name=f"results/alora_async_poisson_{LAMBDA}rps_prompt_len_512.txt", manually_timed_eval_latencies=eval_latencies)


###################################################################

if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(main_poisson())