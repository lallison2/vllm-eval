import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
lambdas = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000]

# component = "eval"
# component = "gen_1"
# component = "gen_2"
component = "gen+eval"

component_title = {'eval': 'Evaluation', 'gen_1': 'First Generation', 'gen_2': 'Second Generation'}

def extract_metrics_from_files(target_metric, is_alora=False, path_prefix="", path_suffix=""):
    metric_vals = []
    # for p_len in prompt_lens:
    # for g_len in gen_lens:
    for LAMBDA in lambdas:
        # file_name = path_prefix + f'alora_prompt_len_{p_len}_{component}.txt' if is_alora else path_prefix + f'lora_prompt_len_{p_len}_{component}.txt'
        # file_name = path_prefix + f'alora_gen_len_{g_len}_{component}.txt' if is_alora else path_prefix + f'lora_gen_len_{g_len}_{component}.txt'
        file_name = path_prefix + f'alora_async_poisson_{LAMBDA}rps{path_suffix}.txt' if is_alora else path_prefix + f'lora_async_poisson_{LAMBDA}rps{path_suffix}.txt'

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
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"End-to-end Latency Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/e2e_latency_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Time-to-first-token Latency Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/ttft_latency_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Request Queue Time Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/queue_time_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Request Inference Time Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/inference_time_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Request Prefill Time Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/prefill_time_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(lambdas, 
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#cdb38f',
            markeredgecolor='#cdb38f',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#f4c5b5",
            markerfacecolor='none',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Request Decode Time Comparison (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/decode_time_arrival_rate_poisson-{component}.png")

    ###############################################
    ###############################################
    ###############################################

    target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of End-to-end Latency (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/e2e_latency_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Time-to-first-token Latency (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/ttft_latency_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Queue Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/queue_time_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Inference Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/inference_time_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Prefill Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/prefill_time_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_arrival_rate/fixed_batch_size/")

    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='meta-llama/Llama-3.3-70B-Instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#f4c5b5",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (rps)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Decode Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/decode_time_speedup_factor_arrival_rate_poisson-{component}.png")

    ###############################################
    ###############################################
    ###############################################

    target_metric = "manually_timed_eval_latency_avg"
    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            llama_alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            llama_lora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#cdb38f",
            markerfacecolor='#f4c5b5',
            markeredgecolor='#f4c5b5',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (requests / s)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Average End-to-end Latency Comparison (Evaluation only) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig("plots/e2e_latency_arrival_rate_poisson-eval.png")

    ###############################################

    target_metric = "manually_timed_eval_latency_avg"
    llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")
    llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/", path_suffix="_llama")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#87C39F",
            markerfacecolor='#87C39F',
            markeredgecolor='#87C39F',
            )
    
    ax.grid(
        axis='x',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )
    ax.grid(
        axis='y',
        which='major',
        linestyle='-',
        linewidth=0.5,
        color='gray',
        alpha=0.7,
    )

    ax.set_xlabel("Arrival Rate (requests / s)")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup of Average End-to-end Latency (Evaluation only) (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig("plots/e2e_latency_speedup_factor_arrival_rate_poisson-eval.png")