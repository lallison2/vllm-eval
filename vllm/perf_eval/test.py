import requests
from openai import OpenAI
import time
import os
import torch
from huggingface_hub import snapshot_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

BASE_NAME = "ibm-granite/granite-3.2-8b-instruct"
ALORA_NAME = "new_alora" #"ibm-granite/granite-3.2-8b-alora-uncertainty"
invocation_string = "<|start_of_role|>certainty<|end_of_role|>"

os.environ["VLLM_USE_V1"] = "1"

# # download your LoRA adapter to ~/.cache/huggingface/…
# alora_path = snapshot_download(repo_id=ALORA_NAME)

# print(alora_path)

###################################################################
prompts = [
    (
        "<|start_of_role|>user<|end_of_role|>What is MIT?<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    ),
]
# Base model call
outputs_base = client.completions.create(model=BASE_NAME,
                                         prompt=prompts,  
                                         temperature=0, 
                                         max_tokens=600)

choices = outputs_base.choices
generated_text = []
for i in range(len(prompts)):
    prompt = prompts[i]

    generated_text += [outputs_base.choices[i].text]
    print(f"Prompt: {prompt!r}, Generated text: {generated_text[-1]!r}")

prompts_alora = [x + y + "<|end_of_text|>\n"+ invocation_string for x,y in zip(prompts, generated_text)] 

# Base model with aLoRA call
t0 = time.time()
alora_outputs = client.completions.create(model=ALORA_NAME,
                                          prompt=prompts_alora, 
                                          temperature=0, 
                                          max_tokens=10)
t = time.time() -t0
print(f"Time: {t}")
for i in range(len(prompts_alora)):
    prompt = prompts_alora[i]
    generated_text = alora_outputs.choices[i].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

###################################################################

stats = ["vllm:kv_cache_usage",
        #  "vllm:prefix_cache_queries",
        #  "vllm:prefix_cache_hits",
        #  "vllm:prompt_tokens",
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

stat_vals = {}
histogram_vals = {}

# Get current Prometheus metrics
metrics = requests.get("http://localhost:8000/metrics/vllm:num_requests_running").text
for line in metrics.splitlines():
    for stat in stats:
        if line.startswith(stat):
            stat_vals[stat] = line.split("}")[-1].strip()
            break

for stat in stats:
    print(stat, stat_vals[stat])