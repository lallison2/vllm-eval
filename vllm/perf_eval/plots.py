import numpy as np
import matplotlib.pyplot as plt

prompt_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
gen_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
lambdas = [0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 50000]

component = "eval"
# component = "gen_1"
# component = "gen_2"
# component = "gen+eval"

component_title = {'eval': 'Evaluation', 'gen_1': 'First Generation', 'gen_2': 'Second Generation'}

def extract_metrics_from_files(target_metric, is_alora=False, path_prefix="", path_suffix="", component=component, varying_comp="gen_len"):
    metric_vals = []

    varying_list = gen_lens
    if varying_comp == "prompt_len":
        varying_list = prompt_lens
    elif varying_comp == "async_poisson":
        varying_list = lambdas

    for value in varying_list:
        file_name = path_prefix + f'alora_{varying_comp}_{value}_{component}{path_suffix}.txt' if is_alora else path_prefix + f'lora_{varying_comp}_{value}_{component}{path_suffix}.txt'
        # file_name = path_prefix + f'alora_{varying_comp}_{value}rps{path_suffix}.txt' if is_alora else path_prefix + f'lora_{varying_comp}_{value}rps{path_suffix}.txt'

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
    
    # target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    # # granite_alora_metric_vals_gen_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="gen_1")
    # # granite_lora_metric_vals_gen_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="gen_1")
    
    # # granite_alora_metric_vals_eval = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="eval")
    # # granite_lora_metric_vals_eval = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="eval")

    # # granite_alora_metric_vals_gen_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="gen_2")
    # # granite_lora_metric_vals_gen_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_arrival_rate_fixed_batch/5_adapters/", path_suffix="_granite_5_adapters", component="gen_2")

    # # granite_alora_metric_vals = [a + b + c for a, b, c in zip(granite_alora_metric_vals_gen_1, granite_alora_metric_vals_eval, granite_alora_metric_vals_gen_2)]
    # # granite_lora_metric_vals = [a + b + c for a, b, c in zip(granite_lora_metric_vals_gen_1, granite_lora_metric_vals_eval, granite_lora_metric_vals_gen_2)]

    # # target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")

    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation Latency Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_e2e_latency_gen_len-{component}.png")


    # ###############################################

    # target_metric = "vllm:time_to_first_token_seconds_sum"

    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")

    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")    

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")


    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(prompt_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation TTFT Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_ttft_latency_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_queue_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation Queue Time Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_queue_time_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_inference_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation Inference Time Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_inference_time_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_prefill_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation Prefill Time Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_prefill_time_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_decode_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # alora_std = []
    # # for i in range(len(alora_mean)):
    # #     std = (granite_alora_metric_vals_trial_1[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_2[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_3[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_4[i] - alora_mean[i]) ** 2
    # #     std += (granite_alora_metric_vals_trial_5[i] - alora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     alora_std.append(std)
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # lora_std = []
    # # for i in range(len(lora_mean)):
    # #     std = (granite_lora_metric_vals_trial_1[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_2[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_3[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_4[i] - lora_mean[i]) ** 2
    # #     std += (granite_lora_metric_vals_trial_5[i] - lora_mean[i]) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     lora_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         alora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(alora_mean, alora_std)], [a - b for a, b in zip(alora_mean, alora_std)], color="#9AA3FF", alpha=0.4)
    # # ax.plot(gen_lens, 
    # #         lora_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#6675A9",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(lora_mean, lora_std)], [a - b for a, b in zip(lora_mean, lora_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct (rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407 (rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )

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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title(f"Evaluation Decode Time Comparison (Base-Adapter)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_decode_time_gen_len-{component}.png")

    # ###############################################
    # ###############################################
    # ###############################################

    # target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_e2e_latency_speedup_factor_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:time_to_first_token_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation TTFT Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_ttft_latency_speedup_factor_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_queue_time_seconds_sum"

    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation Queue Time Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.yaxis.get_offset_text().set_fontsize(16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_queue_time_speedup_factor_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_inference_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation Inference Time Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_inference_time_speedup_factor_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_prefill_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation Prefill Time Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_prefill_time_speedup_factor_gen_len-{component}.png")

    # ###############################################

    # target_metric = "vllm:request_decode_time_seconds_sum"
    
    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # # alora_mean = [(granite_alora_metric_vals_trial_1[i] + granite_alora_metric_vals_trial_2[i]+ granite_alora_metric_vals_trial_3[i] + granite_alora_metric_vals_trial_4[i] + granite_alora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_alora_metric_vals_trial_1))]
    # # lora_mean = [(granite_lora_metric_vals_trial_1[i] + granite_lora_metric_vals_trial_2[i]+ granite_lora_metric_vals_trial_3[i] + granite_lora_metric_vals_trial_4[i] + granite_lora_metric_vals_trial_5[i]) / 5.0 for i in range(len(granite_lora_metric_vals_trial_1))]
    # # ratio_mean = []
    # # ratio_std = []
    # # for i in range(len(alora_mean)):
    # #     avg_ratio = (granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i])
    # #     avg_ratio += (granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i])
    # #     avg_ratio /= 5.0
    # #     ratio_mean.append(avg_ratio)

    # #     std = ((granite_lora_metric_vals_trial_1[i] / granite_alora_metric_vals_trial_1[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_2[i] / granite_alora_metric_vals_trial_2[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_3[i] / granite_alora_metric_vals_trial_3[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_4[i] / granite_alora_metric_vals_trial_4[i]) - avg_ratio) ** 2
    # #     std += ((granite_lora_metric_vals_trial_5[i] / granite_alora_metric_vals_trial_5[i]) - avg_ratio) ** 2
    # #     std /= (5 - 1)
    # #     std = np.sqrt(std)
    # #     ratio_std.append(std)

    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # # ax.plot(gen_lens, 
    # #         ratio_mean, 
    # #         label='ibm-granite/granite-3.2-8b-instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#6675A9",
    # #         markerfacecolor='#6675A9',
    # #         markeredgecolor='#6675A9',
    # #         )
    # # ax.fill_between(gen_lens, [a + b for a, b in zip(ratio_mean, ratio_std)], [a - b for a, b in zip(ratio_mean, ratio_std)], color="#9AA3FF", alpha=0.4)

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title(f"Evaluation Decode Time Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_decode_time_speedup_factor_gen_len-{component}.png")

    ##############################################
    ##############################################
    ##############################################

    # target_metric = "manually_timed_eval_latency_avg"

    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(6, 8))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens,
    #         granite_alora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct\n(rank-32 aLoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )
    # ax.plot(gen_lens,
    #         granite_lora_metric_vals, 
    #         label='ibm-granite/granite-3.2-8b-instruct\n(rank-8 LoRA)', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         llama_alora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct\n(rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    # # ax.plot(gen_lens, 
    # #         llama_lora_metric_vals, 
    # #         label='meta-llama/Llama-3.3-70B-Instruct\n(rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#cdb38f",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         mistral_alora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407\n(rank-32 aLoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    # # ax.plot(gen_lens, 
    # #         mistral_lora_metric_vals, 
    # #         label='mistralai/Mistral-Large-Instruct-2407\n(rank-8 LoRA)', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle=':',
    # #         color="#77a988",
    # #         markerfacecolor='none',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Latency (s)", fontsize=16)
    # ax.set_title("Evaluation Latency Comparison (Base-Adapter", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig("plots/base_adapter_e2e_latency_gen_len-eval.png")

    # ###############################################

    # target_metric = "manually_timed_eval_latency_avg"

    # # granite_alora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_lora_metric_vals_trial_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_1")
    # # granite_alora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_lora_metric_vals_trial_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_2")
    # # granite_alora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_lora_metric_vals_trial_3 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_3")
    # # granite_alora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_lora_metric_vals_trial_4 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_4")
    # # granite_alora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    # # granite_lora_metric_vals_trial_5 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite", path_suffix="_granite_trial_5")
    
    # granite_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")

    # # llama_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")
    # # llama_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_arrival_rate/")

    # # mistral_alora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")
    # # mistral_lora_metric_vals = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_arrival_rate/")

    # fig, ax = plt.subplots(figsize=(6, 8))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens,
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         label='ibm-granite/granite-3.2-8b-instruct', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         )

    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(llama_lora_metric_vals, llama_alora_metric_vals)], 
    # #         label='meta-llama/Llama-3.3-70B-Instruct', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#cdb38f",
    # #         markerfacecolor='#cdb38f',
    # #         markeredgecolor='#cdb38f',
    # #         )
    
    # # ax.plot(gen_lens, 
    # #         [a / b for a, b in zip(mistral_lora_metric_vals, mistral_alora_metric_vals)], 
    # #         label='mistralai/Mistral-Large-Instruct-2407', 
    # #         marker='D', 
    # #         markersize=4,
    # #         linestyle='-',
    # #         color="#77a988",
    # #         markerfacecolor='#77a988',
    # #         markeredgecolor='#77a988',
    # #         )
    
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

    # ax.set_xlabel("Generation Length", fontsize=16)
    # ax.set_ylabel("Speedup", fontsize=16)
    # ax.set_title("Evaluation Speedup\n(Base-Adapter) (LoRA / aLoRA)", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig("plots/base_adapter_e2e_latency_speedup_factor_gen_len-eval.png")

    ###############################################
    ###############################################
    ###############################################

    # target_metric = "vllm:prefix_cache_hits_total"
    
    # granite_alora_metric_vals_gen_len = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter/varying_gen_len/fixed_batch_size/post-fix/", path_suffix="_granite")
    # granite_alora_metric_vals_prompt_len = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/base_adapter/varying_prompt_len/fixed_batch_size/", path_suffix="")
    
    # fig, ax = plt.subplots(figsize=(12, 5))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # num_activation_tokens = 4
    # num_eot_tokens = 2
    # num_eval_tokens = 16
    
    # gen_len_trial_batch_size = 351104 // (256 + gen_lens[7] + num_eot_tokens + num_activation_tokens + num_eval_tokens)
    # ax.plot([256 + g_len + num_eot_tokens + num_activation_tokens + num_eval_tokens for g_len in gen_lens], 
    #         [metric / gen_len_trial_batch_size for metric in granite_alora_metric_vals_gen_len], 
    #         label='# of cache hits when varying generation length', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#A97777",
    #         markerfacecolor='#A97777',
    #         markeredgecolor='#A97777',
    #         linewidth=3,
    #         )
    
    # prompt_len_trial_batch_size = 351104 // (prompt_lens[9] + 256 + num_eot_tokens + num_activation_tokens + num_eval_tokens)
    # ax.plot([p_len + 256 + num_eot_tokens + num_activation_tokens + num_eval_tokens for p_len in prompt_lens][:8], 
    #         [metric / prompt_len_trial_batch_size for metric in granite_alora_metric_vals_prompt_len[:8]], 
    #         label='# of cache hits when varying prompt length', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#88A9A9",
    #         markerfacecolor='#88A9A9',
    #         markeredgecolor='#88A9A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Total Sequence Length", fontsize=16)
    # ax.set_ylabel("# of Cache Hits / Request", fontsize=22)
    # ax.set_title(f"Cache Hit Comparison", fontsize=16)
    # ax.legend(fontsize=10, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # plt.tight_layout()

    # plt.savefig(f"plots/base_adapter_cache_hit-{component}.png")

    #######################################
    #######################################
    #######################################


    # target_metric = "vllm:e2e_request_latency_seconds_sum"
    
    # granite_alora_metric_vals_gen_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_1")
    # granite_lora_metric_vals_gen_1 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_1")
    
    # granite_alora_metric_vals_eval = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="eval")
    # granite_lora_metric_vals_eval = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="eval")

    # granite_alora_metric_vals_gen_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_2")
    # granite_lora_metric_vals_gen_2 = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_2")

    # granite_alora_metric_vals = [a + b + c for a, b, c in zip(granite_alora_metric_vals_gen_1, granite_alora_metric_vals_eval, granite_alora_metric_vals_gen_2)]
    # granite_lora_metric_vals = [a + b + c for a, b, c in zip(granite_lora_metric_vals_gen_1, granite_lora_metric_vals_eval, granite_lora_metric_vals_gen_2)]

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens, 
    #         granite_alora_metric_vals, 
    #         label='rank-32 aLoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
    #         )
    # ax.plot(gen_lens, 
    #         granite_lora_metric_vals, 
    #         label='rank-8 LoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Latency (s)", fontsize=20)
    # ax.set_title(f"E2E Latency Comparison\n(Base-Adapter-Base)", fontsize=20)
    # ax.legend(fontsize=16, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig(f"plots/base_adapter_base_e2e_latency_gen_len-all.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # ax.set_ylim(1.0, 1.2)

    # ax.plot(gen_lens, 
    #         [a / b for a, b in zip(granite_lora_metric_vals, granite_alora_metric_vals)], 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Speedup", fontsize=20)
    # ax.set_title("E2E Speedup\n(Base-Adapter-Base) (LoRA / aLoRA)", fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig("plots/base_adapter_base_e2e_latency_speedup_factor_gen_len-all.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens, 
    #         granite_alora_metric_vals_gen_2, 
    #         label='rank-32 aLoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
    #         )
    # ax.plot(gen_lens, 
    #         granite_lora_metric_vals_gen_2, 
    #         label='rank-8 LoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Latency (s)", fontsize=20)
    # ax.set_title(f"Second Base Call Latency Comparison\n(Base-Adapter-Base)", fontsize=20)
    # ax.legend(fontsize=16, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig(f"plots/base_adapter_base_e2e_latency_gen_len-gen_2.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # ax.set_ylim(1.0, 22.0)

    # ax.plot(gen_lens, 
    #         [a / b for a, b in zip(granite_lora_metric_vals_gen_2, granite_alora_metric_vals_gen_2)],  
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Speedup", fontsize=20)
    # ax.set_title("Second Base Call Speedup\n(Base-Adapter-Base) (LoRA / aLoRA)", fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig("plots/base_adapter_base_e2e_latency_speedup_factor_gen_len-gen_2.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens, 
    #         granite_alora_metric_vals_eval, 
    #         label='rank-32 aLoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
    #         )
    # ax.plot(gen_lens, 
    #         granite_lora_metric_vals_eval, 
    #         label='rank-8 LoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Latency (s)", fontsize=20)
    # ax.set_title(f"Adapter Evaluation Latency Comparison\n(Base-Adapter-Base)", fontsize=20)
    # ax.legend(fontsize=16, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig(f"plots/base_adapter_base_e2e_latency_gen_len-eval.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # ax.set_ylim(1.0, 25.0)

    # ax.plot(gen_lens, 
    #         [a / b for a, b in zip(granite_lora_metric_vals_eval, granite_alora_metric_vals_eval)],  
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Speedup", fontsize=20)
    # ax.set_title("Adapter Evaluation Speedup\n(Base-Adapter-Base) (LoRA / aLoRA)", fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig("plots/base_adapter_base_e2e_latency_speedup_factor_gen_len-eval.png")

    # #######################################

    # granite_alora_metric_vals_eval_gen_2 = [b + c for b, c in zip(granite_alora_metric_vals_eval, granite_alora_metric_vals_gen_2)]
    # granite_lora_metric_vals_eval_gen_2 = [b + c for b, c in zip(granite_lora_metric_vals_eval, granite_lora_metric_vals_gen_2)]

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens, 
    #         granite_alora_metric_vals_eval_gen_2, 
    #         label='rank-32 aLoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
    #         )
    # ax.plot(gen_lens, 
    #         granite_lora_metric_vals_eval_gen_2, 
    #         label='rank-8 LoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Latency (s)", fontsize=20)
    # ax.set_title(f"Eval + Second Base Call Latency Comparison\n(Base-Adapter-Base)", fontsize=20)
    # ax.legend(fontsize=16, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig(f"plots/base_adapter_base_e2e_latency_gen_len-eval+gen2.png")

    # #######################################

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    # ax.set_ylim(1.0, 25.0)

    # ax.plot(gen_lens, 
    #         [a / b for a, b in zip(granite_lora_metric_vals_eval_gen_2, granite_alora_metric_vals_eval_gen_2)], 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Speedup", fontsize=20)
    # ax.set_title("Eval + Second Base Call Speedup\n(Base-Adapter-Base) (LoRA / aLoRA)", fontsize=20)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig("plots/base_adapter_base_e2e_latency_speedup_factor_gen_len-eval+gen_2.png")

    # #######################################

    # target_metric = "vllm:request_queue_time_seconds_sum"

    # granite_alora_metric_vals_gen_2_queue = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=True, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_2")
    # granite_lora_metric_vals_gen_2_queue = extract_metrics_from_files(target_metric, varying_comp="gen_len", is_alora=False, path_prefix="results/base_adapter_base/varying_gen_len_fixed_batch/5_adapters/post-fix/", path_suffix="_granite_5_adapters", component="gen_2")

    # fig, ax = plt.subplots(figsize=(9, 7))

    # from matplotlib.ticker import LogLocator, LogFormatterMathtext
    # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # ax.plot(gen_lens, 
    #         granite_alora_metric_vals_gen_2_queue, 
    #         label='rank-32 aLoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle='-',
    #         color="#6675A9",
    #         markerfacecolor='#6675A9',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
    #         )
    # ax.plot(gen_lens, 
    #         granite_lora_metric_vals_gen_2_queue, 
    #         label='rank-8 LoRA', 
    #         marker='D', 
    #         markersize=4,
    #         linestyle=':',
    #         color="#6675A9",
    #         markerfacecolor='none',
    #         markeredgecolor='#6675A9',
    #         linewidth=3,
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

    # ax.set_xlabel("Generation Length", fontsize=20)
    # ax.set_ylabel("Latency (s)", fontsize=20)
    # ax.set_title(f"Second Base Call Queue Time Comparison\n(Base-Adapter-Base)", fontsize=20)
    # ax.legend(fontsize=16, markerscale=1.0)
    # ax.tick_params(axis='both', which='major', labelsize=20)

    # plt.savefig(f"plots/base_adapter_base_queue_time_gen_len-gen2.png")

    #################################
    #################################
    #################################

    target_metric = "vllm:iteration_tokens_total_sum"

    granite_alora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/base_adapter/varying_prompt_len/fixed_batch_size/", path_suffix="")
    granite_lora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/base_adapter/varying_prompt_len/fixed_batch_size/", path_suffix="")

    llama_alora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_prompt_len/")
    llama_lora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_prompt_len/")

    mistral_alora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_prompt_len/")
    mistral_lora_metric_vals_tokens = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_prompt_len/")

    target_metric = "vllm:e2e_request_latency_seconds_sum"

    granite_alora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/base_adapter/varying_prompt_len/fixed_batch_size/", path_suffix="")
    granite_lora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/base_adapter/varying_prompt_len/fixed_batch_size/", path_suffix="")

    llama_alora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/multi_gpu/llama_3.3_70b/varying_prompt_len/")
    llama_lora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/multi_gpu/llama_3.3_70b/varying_prompt_len/")

    mistral_alora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=True, path_prefix="results/multi_gpu/mistral_large/varying_prompt_len/")
    mistral_lora_metric_vals_e2e = extract_metrics_from_files(target_metric, varying_comp="prompt_len", is_alora=False, path_prefix="results/multi_gpu/mistral_large/varying_prompt_len/")


    models = ('Granite\n3.2 8b', 'Llama\n3.3 70B', 'Mistral\nLarge 2')
    adapter_type = {
        'vanilla vLLM + LoRA': (granite_lora_metric_vals_tokens[9] / granite_lora_metric_vals_e2e[9], llama_lora_metric_vals_tokens[9] / llama_lora_metric_vals_e2e[9], mistral_lora_metric_vals_tokens[9] / mistral_lora_metric_vals_e2e[9]),
        'modified vLLM + aLoRA': (granite_alora_metric_vals_tokens[9] / granite_alora_metric_vals_e2e[9], llama_alora_metric_vals_tokens[9] / llama_alora_metric_vals_e2e[9], mistral_alora_metric_vals_tokens[9] / mistral_alora_metric_vals_e2e[9]),
    }
    bar_color = '#A97777'
    fill_order = [False, True]
    hatch_order = ['/', '']

    print("lora: ", (granite_lora_metric_vals_tokens[9] / granite_lora_metric_vals_e2e[9], llama_lora_metric_vals_tokens[9] / llama_lora_metric_vals_e2e[9], mistral_lora_metric_vals_tokens[9] / mistral_lora_metric_vals_e2e[9]))
    print("alora: ", (granite_alora_metric_vals_tokens[9] / granite_alora_metric_vals_e2e[9], llama_alora_metric_vals_tokens[9] / llama_alora_metric_vals_e2e[9], mistral_alora_metric_vals_tokens[9] / mistral_alora_metric_vals_e2e[9]))
    
    x = np.arange(3)
    width = 0.25  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

    from matplotlib.ticker import LogLocator, LogFormatterMathtext
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    for attribute, measurement in adapter_type.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=bar_color, fill=fill_order[multiplier], hatch=hatch_order[multiplier], edgecolor=bar_color)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel('Throughput (tokens/s)', fontsize=20)
    ax.set_title('Throughput Comparison (Prompt Length = 65536)', fontsize=20)
    ax.set_xticks(x + width, models)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.35),  # below plots but inside figure
        ncol=2,
        frameon=True,
        fontsize=16, markerscale=1.0,
    )
    fig.subplots_adjust(bottom=0.1)

    plt.savefig(f"plots/throughput_comparison.png")