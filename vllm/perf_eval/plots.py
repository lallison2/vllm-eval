import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

def extract_metrics_from_files(target_metric, is_alora=False):
    metric_vals = []
    for p_len in prompt_lens:
        file_name = f'results/alora_prompt_len_{p_len}_eval.txt' if is_alora else f'results/lora_prompt_len_{p_len}_eval.txt'

        batch_size = (351104 // (p_len + 2 + 256 + 4 + 16))
        with open(file_name, 'r') as f:
            for line in f:
                prompt_strings = line.strip().split(' ')
                assert len(prompt_strings) == 2, "formatting error reading in results data"
                if prompt_strings[0].startswith(target_metric):
                    value = float(prompt_strings[1])
                    metric_vals.append(value / batch_size)
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

    ax.plot(prompt_lens, 
            alora_metric_vals, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.plot(prompt_lens, 
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

    ax.set_xlabel("Prompt Length")
    ax.set_ylabel("Latency (s)")
    ax.set_title("End-to-end Latency Comparison")
    ax.legend(fontsize=8, markerscale=0.7)

    plt.savefig("plots/e2e_latency_prompt_len_eval.png")

    ###############################################

    # target_metric = "vllm:time_to_first_token_seconds_sum"
    # alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    # lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(prompt_lens, 
    #         alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(prompt_lens, 
    #         lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.grid(
    #     axis='x',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )
    # ax.grid(
    #     axis='y',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )

    # ax.set_xlabel("Prompt Length")
    # ax.set_ylabel("Latency (s)")
    # ax.set_title("Time-to-first-token Latency Comparison")
    # ax.legend(fontsize=8, markerscale=0.7)

    # plt.savefig("plots/ttft_latency_prompt_len_eval.png")

    ###############################################

    # target_metric = "vllm:request_queue_time_seconds_sum"
    # alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    # lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(prompt_lens, 
    #         alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(prompt_lens, 
    #         lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.grid(
    #     axis='x',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )
    # ax.grid(
    #     axis='y',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )

    # ax.set_xlabel("Prompt Length")
    # ax.set_ylabel("Latency (s)")
    # ax.set_title("Request Queue Time Comparison")
    # ax.legend(fontsize=8, markerscale=0.7)

    # plt.savefig("plots/queue_time_prompt_len_eval.png")

    ###############################################

    # target_metric = "vllm:request_inference_time_seconds_sum"
    # alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    # lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(prompt_lens, 
    #         alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(prompt_lens, 
    #         lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.grid(
    #     axis='x',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )
    # ax.grid(
    #     axis='y',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )

    # ax.set_xlabel("Prompt Length")
    # ax.set_ylabel("Latency (s)")
    # ax.set_title("Request Inference Time Comparison")
    # ax.legend(fontsize=8, markerscale=0.7)

    # plt.savefig("plots/inference_time_prompt_len_eval.png")

    ###############################################

    # target_metric = "vllm:request_prefill_time_seconds_sum"
    # alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    # lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(prompt_lens, 
    #         alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(prompt_lens, 
    #         lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.grid(
    #     axis='x',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )
    # ax.grid(
    #     axis='y',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )

    # ax.set_xlabel("Prompt Length")
    # ax.set_ylabel("Latency (s)")
    # ax.set_title("Request Prefill Time Comparison")
    # ax.legend(fontsize=8, markerscale=0.7)

    # plt.savefig("plots/prefill_time_prompt_len_eval.png")

    ###############################################

    # target_metric = "vllm:request_decode_time_seconds_sum"
    # alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True)
    # lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False)

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(prompt_lens, 
    #         alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(prompt_lens, 
    #         lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.grid(
    #     axis='x',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )
    # ax.grid(
    #     axis='y',
    #     which='major',
    #     linestyle='-',
    #     linewidth=0.5,
    #     color='gray',
    #     alpha=0.7,
    # )

    # ax.set_xlabel("Prompt Length")
    # ax.set_ylabel("Latency (s)")
    # ax.set_title("Request Decode Time Comparison")
    # ax.legend(fontsize=8, markerscale=0.7)

    # plt.savefig("plots/decode_time_prompt_len_eval.png")