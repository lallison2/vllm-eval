import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

def extract_metrics_from_files(target_metric, is_alora=False):
    metric_vals = []
    for p_len in prompt_lens:
        file_name = f'results/alora_prompt_len_{p_len}.txt' if is_alora else f'results/lora_prompt_len_{p_len}.txt'

        with open(file_name, 'r') as f:
            for line in f:
                prompt_strings = line.strip().split(' ')
                assert len(prompt_strings) == 2, "formatting error reading in results data"
                if prompt_strings[0].startswith(target_metric):
                    value = float(prompt_strings[1])
                    metric_vals.append(value)
                    break
    return metric_vals
    

if __name__ == '__main__':
    
    target_metric = "vllm:e2e_request_latency_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # for i, p_len in enumerate(prompt_lens):
    #     print(f"prompt len: {p_len}, alora metric val: {alora_metric_vals[i]}, lora metric val: {lora_metric_vals[i]}")

    fig, ax = plt.subplots()

    from matplotlib.ticker import LogFormatterExponent, LogLocator
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterExponent(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterExponent(base=10.0))

    ax.plot(prompt_lens, alora_metric_vals, label='aLoRA')
    ax.plot(prompt_lens, lora_metric_vals, label='LoRA')

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Evaluation Latency Comparison")

    # Display the plot
    plt.savefig("latency_prompt_len.png")