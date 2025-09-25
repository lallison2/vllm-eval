#!/bin/bash

# More documentation: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#vllm-serve
export VLLM_USE_V1="1"

# Specify base model (and optionally loras/aloras) to load in when starting the server.
# If aLoRA, the invocation string field should exist.
# Note: Can load in multiple intrinsics here for easier testing later on.

#### Start server with random alora, granite-3.2-8b ####
# vllm serve ibm-granite/granite-3.2-8b-instruct \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' 
#     # --gpu-memory-utilization 0.7

#### Start server with random lora, granite-3.2-8b ####
# vllm serve ibm-granite/granite-3.2-8b-instruct \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_lora", "path": "/nobackup/users/lallison/hf_cache/hub/random_lora_r_8", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' 
#     # --gpu-memory-utilization 0.7

#### Start server with random alora, meta-llama/Llama-3.3-70B-Instruct ####
vllm serve /nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
    --enable-lora \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --enable-prefix-caching \
    --enable-activated-lora \
    --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "meta-llama/Llama-3.3-70B-Instruct"}' 
    # --gpu-memory-utilization 0.7

# Check that the lora model is listed along with other models.
#curl localhost:8000/v1/models | jq .

