import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

class ALoraConfig(LoraConfig):
    def __init__(self, invocation_string=None, **kwargs):
        super().__init__(**kwargs)
        self.invocation_string = invocation_string

torch.manual_seed(40)
base_model_name = "/nobackup/users/lallison/hf_cache/models--ibm-granite--granite-3.2-8b-instruct/snapshots/610d8c6ee9c84ce51f6dfd7bc5c0215d95d49695"
alora_save_path = "/nobackup/users/lallison/hf_cache/hub/random_alora_r_32_5"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
alora_config = ALoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    invocation_string="<|start_of_role|>certainty<|end_of_role|>",
)
alora_model = get_peft_model(base_model, alora_config)

for name, param in alora_model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        param.data = torch.randn_like(param.data) * 0.2
print("random alora weights initialized (rank 32)")

alora_model.save_pretrained(alora_save_path)

#####################################################################

alora_model.unload()
lora_save_path = "/nobackup/users/lallison/hf_cache/hub/random_lora_r_8_5"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
lora_model = get_peft_model(base_model, lora_config)

for name, param in lora_model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        param.data = torch.randn_like(param.data) * 0.2
print("random lora weights initialized (rank 8)")

lora_model.save_pretrained(lora_save_path)