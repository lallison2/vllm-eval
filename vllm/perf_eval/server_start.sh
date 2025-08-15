#!/bin/bash

# More documentation: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#vllm-serve
export VLLM_USE_V1="1"

# Specify base model (and optionally loras/aloras) to load in when starting the server.
# If aLoRA, the invocation string field should exist.
# Note: Can load in multiple intrinsics here for easier testing later on.

#### Start server with random alora ####
vllm serve ibm-granite/granite-3.2-8b-instruct \
    --enable-lora \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --enable-prefix-caching \
    --enable-activated-lora \
    --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' 
    # --gpu-memory-utilization 0.7
#--no-enable-prefix-caching

#### Start server with random lora ####
# vllm serve ibm-granite/granite-3.2-8b-instruct \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --lora-modules '{"name": "random_lora", "path": "/nobackup/users/lallison/hf_cache/hub/random_lora_r_8", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' 
#     # --gpu-memory-utilization 0.7

# Check that the lora model is listed along with other models.
#curl localhost:8000/v1/models | jq .

