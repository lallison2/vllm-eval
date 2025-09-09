import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    target_metric = "vllm:e2e_request_latency_seconds_sum"
    prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    metric_vals = []
    for p_len in prompt_lens:
        with open(f'results/alora_prompt_len_{p_len}.txt', 'r') as f:
            for line in f:
                prompt_strings = line.strip().split(' ')
                assert len(prompt_strings) == 2, "formatting error reading in results data"
                if prompt_strings[0].startswith(target_metric):
                    value = float(prompt_strings[1])
                    metric_vals.append(value)
                    break
    
    for i, p_len in enumerate(prompt_lens):
        print(f"prompt len: {p_len}, metric val: {metric_vals[i]}")