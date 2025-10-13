#!/bin/bash

# More documentation: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#vllm-serve
export VLLM_USE_V1="1"
export VLLM_LOG_LEVEL=debug

# Specify base model (and optionally loras/aloras) to load in when starting the server.
# If aLoRA, the invocation string field should exist.

# #### Start server with random alora, granite-3.2-8b ####
vllm serve /nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695 \
    --enable-lora \
    --port 8000 \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --enable-prefix-caching \
    --enable-activated-lora \
    --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}' \
    '{"name": "random_alora_2", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32_2", "base_model_name": "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}' \
    '{"name": "random_alora_3", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32_3", "base_model_name": "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}' \
    '{"name": "random_alora_4", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32_4", "base_model_name": "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}' \
    '{"name": "random_alora_5", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32_5", "base_model_name": "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"}' \

#### Start server with random lora, granite-3.2-8b ####
# vllm serve /nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695 \
#     --enable-lora \
#     --port 8000 \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules random_lora=/nobackup/users/lallison/hf_cache/hub/random_lora_r_8 \
#     random_lora_2=/nobackup/users/lallison/hf_cache/hub/random_lora_r_8_2 \
#     random_lora_3=/nobackup/users/lallison/hf_cache/hub/random_lora_r_8_3 \
#     random_lora_4=/nobackup/users/lallison/hf_cache/hub/random_lora_r_8_4 \
#     random_lora_5=/nobackup/users/lallison/hf_cache/hub/random_lora_r_8_5

###########################################################################


# ### Start server with random alora, meta-llama/Llama-3.3-70B-Instruct ####
# vllm serve /nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
#     --tensor-parallel-size 4 \
#     --port 8000 \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "/nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"}' 

## Start server with random lora, meta-llama/Llama-3.3-70B-Instruct ####
# vllm serve /nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
#     --tensor-parallel-size 4 \
#     --port 8000 \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_lora", "path": "/nobackup/users/lallison/hf_cache/hub/random_lora_r_8", "base_model_name": "/nobackup/users/lallison/hf_cache/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"}' 


###########################################################################


# ### Start server with random alora, mistralai/Mistral-Large-Instruct-2407 ####
# vllm serve /nobackup/users/lallison/hf_cache/models--mistralai--Mistral-Large-Instruct-2407/snapshots/a286006d554cb37a61d13c7ae61bc90cc1d372fc \
#     --tensor-parallel-size 8 \
#     --port 8000 \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_alora", "path": "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32", "base_model_name": "/nobackup/users/lallison/hf_cache/models--mistralai--Mistral-Large-Instruct-2407/snapshots/a286006d554cb37a61d13c7ae61bc90cc1d372fc"}' 

## Start server with random lora, mistralai/Mistral-Large-Instruct-2407 ####
# vllm serve /nobackup/users/lallison/hf_cache/models--mistralai--Mistral-Large-Instruct-2407/snapshots/a286006d554cb37a61d13c7ae61bc90cc1d372fc \
#     --tensor-parallel-size 8 \
#     --port 8000 \
#     --enable-lora \
#     --dtype bfloat16 \
#     --max-lora-rank 64 \
#     --enable-prefix-caching \
#     --enable-activated-lora \
#     --lora-modules '{"name": "random_lora", "path": "/nobackup/users/lallison/hf_cache/hub/random_lora_r_8", "base_model_name": "/nobackup/users/lallison/hf_cache/models--mistralai--Mistral-Large-Instruct-2407/snapshots/a286006d554cb37a61d13c7ae61bc90cc1d372fc"}'


# Check that the lora model is listed along with other models.
# curl localhost:8000/v1/models | jq .

