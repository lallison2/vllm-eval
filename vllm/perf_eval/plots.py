import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
lambdas = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000]

component = "eval"
# component = "gen_1"
# component = "gen_2"
# component = "gen+eval"

component_title = {'eval': 'Evaluation', 'gen_1': 'First Generation', 'gen_2': 'Second Generation'}

def extract_metrics_from_files(target_metric, is_alora=False, path_prefix="", path_suffix="", component=component):
    metric_vals = []
#     for p_len in prompt_lens:
    for g_len in gen_lens:
    # for LAMBDA in lambdas:
        # file_name = path_prefix + f'alora_prompt_len_{p_len}_{component}.txt' if is_alora else path_prefix + f'lora_prompt_len_{p_len}_{component}.txt'
        file_name = path_prefix + f'alora_gen_len_{g_len}_{component}{path_suffix}.txt' if is_alora else path_prefix + f'lora_gen_len_{g_len}_{component}{path_suffix}.txt'
        # file_name = path_prefix + f'alora_async_poisson_{LAMBDA}rps{path_suffix}.txt' if is_alora else path_prefix + f'lora_async_poisson_{LAMBDA}rps{path_suffix}.txt'

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
    
    # target_metric_eval = "vllm:e2e_request_latency_seconds_sum"
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric_eval, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric_eval, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/")

#     target_metric_eval = "vllm:e2e_request_latency_seconds_sum"
    
#     granite_alora_metric_vals_eval = extract_metrics_from_files(target_metric_eval, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/", component="eval")
#     granite_lora_metric_vals_eval = extract_metrics_from_files(target_metric_eval, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/", component="eval")

#     target_metric_gen_2 = "vllm:request_prefill_time_seconds_sum"

#     granite_alora_metric_vals_gen_2 = extract_metrics_from_files(target_metric_gen_2, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/", component="gen_2")
#     granite_lora_metric_vals_gen_2 = extract_metrics_from_files(target_metric_gen_2, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/500_total_requests/", component="gen_2")

#     granite_alora_metric_vals = [a + b for a, b in zip(granite_alora_metric_vals_gen_2, granite_alora_metric_vals_eval)]
#     granite_lora_metric_vals = [a + b for a, b in zip(granite_lora_metric_vals_eval, granite_lora_metric_vals_gen_2)]

    target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")

    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"End-to-End Latency Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_e2e_latency_gen_len-{component}.png")


    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"

    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")

    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")


    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"Time-to-first-token Latency Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_ttft_latency_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"Request Queue Time Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_queue_time_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"Request Inference Time Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_inference_time_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"Request Prefill Time Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_prefill_time_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    alora_std = []
    for i in range(len(alora_mean)):
        std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
        std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        alora_std.append(std)
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    lora_std = []
    for i in range(len(lora_mean)):
        std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
        std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        lora_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            alora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    ax.plot(gen_lens, 
            lora_mean, 
            label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
            marker='D', 
            markersize=4,
            linestyle=':',
            color="#6675A9",
            markerfacecolor='none',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
    #         )

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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Average Latency (s)")
    ax.set_title(f"Request Decode Time Comparison (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_decode_time_gen_len-{component}.png")

    ###############################################
    ###############################################
    ###############################################

    target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of End-to-end Latency (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_e2e_latency_speedup_factor_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:time_to_first_token_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of TTFT Latency (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_ttft_latency_speedup_factor_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_queue_time_seconds_sum"

    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Queue Time (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_queue_time_speedup_factor_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_inference_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Inference Time (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_inference_time_speedup_factor_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_prefill_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Speedup of Request Prefill Time (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_prefill_time_speedup_factor_gen_len-{component}.png")

    ###############################################

    target_metric = "vllm:request_decode_time_seconds_sum"
    
    granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    
    alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    ratio_mean = []
    ratio_std = []
    for i in range(len(alora_mean)):
        avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
        avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
        avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
        avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
        avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
        avg_ratio /= 5.0
        ratio_mean.append(avg_ratio)

        std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
        std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
        std /= (5 - 1)
        std = np.sqrt(std)
        ratio_std.append(std)

    # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    fig, ax = plt.subplots(figsize=(8, 6))

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.plot(gen_lens, 
            ratio_mean, 
            label='ibm-granite/granite-3.2-8b-instruct', 
            marker='D', 
            markersize=4,
            linestyle='-',
            color="#6675A9",
            markerfacecolor='#6675A9',
            markeredgecolor='#6675A9',
            )
    ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

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
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    
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

    ax.set_xlabel("Generation Length")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Average Speedup of Request Decode Time (LoRA / aLoRA) (Base-Adapter)")
    ax.legend(fontsize=8, markerscale=1.0)

    plt.savefig(f"plots/base_adapter_decode_time_speedup_factor_gen_len-{component}.png")

    ###############################################
    ###############################################
    ###############################################

    # target_metric = "manually_timed_eval_latency_avg"

    # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

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

    # ax.plot(lambdas, 
    #         llama_alora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#cdb38f",
    #         markerfacecolor='#cdb38f',
    #         markeredgecolor='#cdb38f',
    #         )
    # ax.plot(lambdas, 
    #         llama_lora_metric_vals, 
    #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#f4c5b5",
    #         markerfacecolor='none',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         mistral_alora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
    #         )
    # ax.plot(lambdas, 
    #         mistral_lora_metric_vals, 
    #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#77a988",
    #         markerfacecolor='none',
    #         markeredgecolor='#77a988',
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

    # ax.set_xlabel("Generation Length")
    # ax.set_ylabel("Average Latency (s)")
    # ax.set_title("Average End-to-end Latency Comparison (Evaluation only)")
    # ax.legend(fontsize=8, markerscale=1.0)

    # plt.savefig("plots/base_adapter_e2e_latency_gen_len-eval.png")

    # ###############################################

    # target_metric = "manually_timed_eval_latency_avg"

    # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_1")
    # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_2")
    # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_3")
    # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_4")
    # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/", path_suffix="_granite_trial_5")
    

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_gen_len/500_reqs/", path_suffix="_llama")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_gen_len/500_reqs/", path_suffix="_mistral")

    # fig, ax = plt.subplots(figsize=(8, 6))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

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

    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    #         label='meta-llama/Llama-3.3-70B-Instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#f4c5b5",
    #         markerfacecolor='#f4c5b5',
    #         markeredgecolor='#f4c5b5',
    #         )
    
    # ax.plot(lambdas, 
    #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    #         label='mistralai/Mistral-Large-Instruct-2407', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#77a988",
    #         markerfacecolor='#77a988',
    #         markeredgecolor='#77a988',
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

    # ax.set_xlabel("Generation Length")
    # ax.set_ylabel("Speedup")
    # ax.set_title("Speedup of Average End-to-end Latency (Evaluation only) (LoRA / aLoRA)")
    # ax.legend(fontsize=8, markerscale=1.0)

    # plt.savefig("plots/base_adapter_e2e_latency_speedup_factor_gen_len-eval.png")