#!/bin/bash

# More documentation: https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html#vllm-serve
export VLLM_USE_V1="1"

# Specify base model (and optionally loras/aloras) to load in when starting the server.
# If aLoRA, the invocation string field should exist.
# Note: Can load in multiple intrinsics here for easier testing later on.
vllm serve ibm-granite/granite-3.2-8b-instruct \
    --enable-lora \
    --lora-modules '{"name": "new_alora", "path": "/nobackup/users/lallison/hf_cache/hub/models--ibm-granite--granite-3.2-8b-alora-uncertainty/snapshots/0d8ce48cdd4280a1e8fc37aa1de07537670ecf21", "base_model_name": "ibm-granite/granite-3.2-8b-instruct"}' \
    --dtype bfloat16 \
    --max-lora-rank 64 \
    --enable-prefix-caching
    --port 8000
#--no-enable-prefix-caching

# Check that the lora model is listed along with other models.
#curl localhost:8000/v1/models | jq .

