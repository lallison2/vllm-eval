import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# lambdas = [50, 100, 500, 1000, 5000, 10000, 20000, 50000]
lambdas = [0.5, 1, 5, 10, 50, 100]

def extract_metrics_from_files(target_metric, is_alora=False):
    metric_vals = []
    # for p_len in prompt_lens:
    # for g_len in gen_lens:
    for LAMBDA in lambdas:
        # file_name = f'results/alora_prompt_len_{p_len}_eval.txt' if is_alora else f'results/lora_prompt_len_{p_len}_eval.txt'
        # file_name = f'results/alora_gen_len_{g_len}_eval.txt' if is_alora else f'results/lora_gen_len_{g_len}_eval.txt'
        file_name = f'results/alora_async_poisson_{LAMBDA}rps.txt' if is_alora else f'results/lora_async_poisson_{LAMBDA}rps.txt'

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

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("End-to-end Latency Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/e2e_latency_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("Time-to-first-token Latency Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/ttft_latency_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("Request Queue Time Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/queue_time_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("Request Inference Time Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/inference_time_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("Request Prefill Time Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/prefill_time_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.set_title("Request Decode Time Comparison (Generation + Evaluation) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/decode_time_async_poisson_gen-eval.png")

    ###############################################
    ###############################################
    ###############################################

    target_metric = "vllm:e2e_request_latency_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of End-to-end Latency (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/e2e_latency_speedup_factor_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of Time-to-first-token Latency (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/ttft_latency_speedup_factor_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of Request Queue Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/queue_time_speedup_factor_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of Request Inference Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/inference_time_speedup_factor_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of Request Prefill Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/prefill_time_speedup_factor_async_poisson_gen-eval.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.set_title("Speedup of Request Decode Time (LoRA / aLoRA) (TOTAL_REQS = 300)")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/decode_time_speedup_factor_async_poisson_gen-eval.png")

    ###############################################
    ###############################################
    ###############################################

    target_metric = "manually_timed_eval_latency_avg"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(lambdas, 
            lora_metric_vals, 
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
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/e2e_latency_async_poisson_eval.png")

    ###############################################

    target_metric = "manually_timed_eval_latency_avg"
    alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(lambdas, 
            [a / b for a, b in zip(lora_metric_vals, alora_metric_vals)], 
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
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/e2e_latency_speedup_factor_async_poisson_eval.png")