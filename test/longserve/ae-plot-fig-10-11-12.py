from matplotlib import pyplot as plt
import matplotlib.axes as mpl_axes
import numpy as np
import os
import argparse
import sys
from typing import Iterator
import dataclasses
from typing import Callable, Optional, Union
from lib_benchmark_serving.metrics import BenchmarkMetrics
from lib_benchmark_serving.structs import load_req_result_list, ReqResult

parser = argparse.ArgumentParser()
parser.add_argument("--ae-id", type=str, default="intlsy")
parser.add_argument("--fig", type=str, required=True)
args = parser.parse_args()
ae_id = args.ae_id
fig_id = args.fig

figs_dir = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}"
EXP_RESULT_ROOT = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}"
analytical_model_path = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}/analytical-model.csv"

def set_xinjin_style(ax: mpl_axes.Axes):
    ax.tick_params('x', direction="in")
    ax.tick_params('y', direction="in")
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)

def save_fig(filename: str):
    if not filename.endswith(".png"):
        filename += ".png"
    plt.savefig(os.path.join(figs_dir, filename), bbox_inches='tight')

def walk_around_exp_results(dataset: str, backend: str) -> Iterator[tuple[str, str, str, int, float]]:
    """
    Walk around all .exp files in EXP_RESULT_ROOT/{dataset}, and
    yield (filepath, filename, backend, num_prompts, req_rate)
    """
    dataset = dataset.lower()
    exp_result_dir = os.path.join(EXP_RESULT_ROOT, dataset)
    candidates = []
    if not os.path.exists(exp_result_dir):
        return
    for filename in os.listdir(exp_result_dir):
        if filename.endswith(".exp") and filename.startswith(backend+'-') and not filename.endswith("-uniform.exp"):
            filename_part = filename[len(backend)+1:-4]
            if filename_part.count('-') != 1:
                continue
            num_prompts = int(filename_part.split('-')[0])
            req_rate = float(filename_part.split('-')[1])
            filepath = os.path.join(exp_result_dir, filename)
            candidates.append((filepath, filename, backend, num_prompts, req_rate))
    
    # Filter out invalid candidates
    # A candidate (n, r) is invalid, if
    # there exists a (n', r') and n' > n, r' <= r
    valid_candidates = []
    for candidate in candidates:
        _, _, _, n, r = candidate
        is_invalid = False
        for (_, _, _, n_, r_) in candidates:
            if r_ <= r and n_ > n:
                is_invalid = True
                break
        if not is_invalid:
            valid_candidates.append(candidate)
        
    valid_candidates.sort(key=lambda x: x[4])	# Sort by req_rate
    for candidate in valid_candidates:
        yield candidate

def load_metric(exp_result_path: str) -> tuple[BenchmarkMetrics, list[ReqResult]]:
    """
    Calculate `BenchmarkMetrics` from the given exp result file
    """
    req_results = load_req_result_list(exp_result_path)
    metrics = BenchmarkMetrics.from_req_results(req_results)
    return metrics, req_results

@dataclasses.dataclass
class PlotRowConfig:
    dataset_name: str
    label_name: str

@dataclasses.dataclass
class PlotColConfig:
    title: str
    retriever: Callable[[BenchmarkMetrics, list[ReqResult], int], float]

@dataclasses.dataclass
class BackendConfig:
    name: str
    label: str
    color: str
    marker: str

def get_intersection_point(
        xs: list[float],
        ys: list[float],
        target_y: float
    ) -> float:
    """
    Get the first intersection point of the line (xs, ys) with y=target_y
    """
    for i in range(len(xs)-1):
        if (ys[i] >= target_y) != (ys[i+1] >= target_y):
            return xs[i] + (xs[i+1]-xs[i]) * (target_y-ys[i]) / (ys[i+1]-ys[i])
    return float("inf")

def plot_one_latency_rate_plot(ax: mpl_axes.Axes, dataset: str, retriever: Callable[[BenchmarkMetrics, list[ReqResult], int], float], backends: list[BackendConfig], plot_row: int, normalizer: float, intersection_y: float = None) -> list[float]:
    intersection_points = []
    for cur_backend in backends:
        cur_xs = []
        cur_ys = []
        for (file_path, _, _, num_prompts, req_rate) in walk_around_exp_results(dataset, cur_backend.name):
            metrics, req_results = load_metric(file_path)
            cur_xs.append(req_rate)
            cur_ys.append(retriever(metrics, req_results, plot_row)/normalizer)
        
        ax.plot(cur_xs, cur_ys, label=cur_backend.label, marker=cur_backend.marker, color=cur_backend.color)
        intersection_points.append(get_intersection_point(cur_xs, cur_ys, intersection_y) if intersection_y is not None else 0.0)
    return intersection_points

def draw_one_row_of_plots(
    fig,
    axs: list[mpl_axes.Axes],
    plot_i: int,
    backends: list[BackendConfig],
    row_configs: list[PlotRowConfig],
    col_configs: list[PlotColConfig],
    y_limits: list[list[Union[int, float]]],
    slos: Optional[list[list[Union[int, float]]]] = None,
    normalize_to_slo: bool = False
):
    row_config = row_configs[plot_i]
    intersect_xs: list[list[float]] = []
    for (plot_j, col_config) in enumerate(col_configs):
        cur_ax = axs[plot_j]
        cur_y_limit = y_limits[plot_i][plot_j] if plot_i < len(y_limits) and plot_j < len(y_limits[plot_i]) else None
        cur_slo = slos[plot_i][plot_j] if slos is not None and plot_i < len(slos) and plot_j < len(slos[plot_i]) else None

        if cur_y_limit is not None:
            cur_ax.set_ylim(0, cur_y_limit)

        if plot_j == 0:
            cur_ax.set_ylabel(row_config.label_name + "\n" + r"${ }^{\mathrm{Norm.\ Latency\ (s/token)}}$", fontsize=18)
        if plot_i == 3:
            cur_ax.set_xlabel("Request Rate (req/s)")

        if normalize_to_slo and cur_slo is None:
            raise ValueError("When normalize_to_slo is True, slos must be provided")
        cur_intersect_xs = plot_one_latency_rate_plot(cur_ax, row_config.dataset_name, col_config.retriever, backends, plot_i, cur_slo if normalize_to_slo else 1.0, 1.0)
        intersect_xs.append(cur_intersect_xs)

        if plot_i == 0 and plot_j == 0:
            fig.legend(frameon=False, loc="upper center", ncol=3, prop={'size': 18}, bbox_to_anchor=(0.5, 0.96))

        if normalize_to_slo:
            cur_ax.axhline(y=1.0, color='grey', linestyle='--', label="SLO")
        elif cur_slo is not None:
            cur_ax.axhline(y=cur_slo, color='grey', linestyle='--', label="SLO")
        cur_ax.set_xlim(left=0)
        set_xinjin_style(cur_ax)
        
    assert backends[0].label == "LoongServe"
    for (backend_i, backend) in enumerate(backends):
        for (plot_j, col_config) in enumerate(col_configs):
            cur_intersect_xs = intersect_xs[plot_j][backend_i]
            longserve_intersect_xs = intersect_xs[plot_j][0]
            speedup = longserve_intersect_xs / cur_intersect_xs
            print(f"{backend.label} {col_config.title} speedup: {speedup}")

def draw_fig_10():
    plt.rcParams.update({"font.size": 16})
    backends = [
        BackendConfig("longserve", "LoongServe", "C0", "o"),
        BackendConfig("vllm", "vLLM", "C1", "s"),
        BackendConfig("deepspeed", "DeepSpeed MII (Dynamic SplitFuse)", "C2", "v"),
        BackendConfig("distserve", "DistServe (Prefill-Decoding Disaggregation)", "C4", "^"),
        BackendConfig("lightllm-sf", "LightLLM w/ SplitFuse", "C3", "D"),
    ]
    row_configs = [
        PlotRowConfig("ShareGPT", "ShareGPT"),
        PlotRowConfig("LEval", "LEval"),
        PlotRowConfig("LV-Eval", "LV-Eval"),
        PlotRowConfig("Mixed1", "Mixed"),
    ]
    col_configs = [
        PlotColConfig("Avg token latency", lambda x, _, row: x.avg_per_token_latency_ms),
        PlotColConfig("Avg input token latency", lambda x, _, row: x.avg_input_token_latency_ms),
        PlotColConfig("Avg output token latency", lambda x, _, row: x.avg_output_token_latency_ms),
    ]

    slos = [
        [92, 20, 320],
        [3.5, 1.8, 320],
        [2.5, 2.5, 365],
        [70, 90, 360]
    ]
    y_limits = [
        [1.2 for x in slos[row]]
        for row in range(len(slos))
    ]
    
    fig, axs = plt.subplots(4, 3, figsize=(20, 15))
    draw_one_row_of_plots(fig, axs[0], 0, backends, row_configs, col_configs, y_limits, slos, normalize_to_slo=True)
    draw_one_row_of_plots(fig, axs[1], 1, backends, row_configs, col_configs, y_limits, slos, normalize_to_slo=True)
    draw_one_row_of_plots(fig, axs[2], 2, backends, row_configs, col_configs, y_limits, slos, normalize_to_slo=True)
    draw_one_row_of_plots(fig, axs[3], 3, backends, row_configs, col_configs, y_limits, slos, normalize_to_slo=True)

    save_fig("fig10.png")

def draw_fig_11():
    plt.rcParams.update({"font.size": 16})
    backends = [
        BackendConfig("longserve-multi-node", "LoongServe", "C0", "o"),
        BackendConfig("vllm-multi-node", "vLLM", "C1", "s"),
        BackendConfig("lightllm-sf-multi-node", "LightLLM w/ SplitFuse", "C3", "D"),
    ]
    row_configs = [
        PlotRowConfig("Mixed1", "Mixed"),
    ]
    col_configs = [
        PlotColConfig("Avg token latency", lambda x, _, row: x.avg_per_token_latency_ms),
        PlotColConfig("Avg input token latency", lambda x, _, row: x.avg_input_token_latency_ms),
        PlotColConfig("Avg output token latency", lambda x, _, row: x.avg_output_token_latency_ms),
    ]

    slos = [
        [65, 64, 200],
    ]
    y_limits = [
        [1.2 for x in slos[row]]
        for row in range(len(slos))
    ]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 3))
    draw_one_row_of_plots(fig, axs, 0, backends, row_configs, col_configs, y_limits, slos, normalize_to_slo=True)

    save_fig("fig11.png")

def draw_fig_12():
    import math
    from lib_benchmark_serving.profiler import Profiler
    profiler = Profiler(analytical_model_path)

    def get_total_slo_attainment(req_results: list[ReqResult], slo: float) -> tuple[float, float]:
        num_valid_tokens = 0
        num_total_tokens = 0
        num_valid_reqs = 0
        num_total_reqs = 0
        for r in req_results:
            num_total_tokens += r.prompt_len + r.output_len - 1
            num_total_reqs += 1
            if r.latency*1000 / (r.prompt_len+r.output_len-1) <= slo:
                num_valid_tokens += r.prompt_len + r.output_len - 1
                num_valid_reqs += 1
        return num_valid_tokens / num_total_tokens, num_valid_reqs / num_total_reqs
        
    def check_is_prefill_latency_valid(req_result: ReqResult, tol: float, slo_lower_bound: float) -> bool:
        normal_time_usage = profiler.predict_time_consumption_one_req(req_result.prompt_len)
        slo = max(normal_time_usage * tol, slo_lower_bound)
        return req_result.ttft*1000 <= slo

    def get_both_slo_attainment(req_results: list[ReqResult], ttft_tol: float, slo_lower_bound: float, tpot_slo: float) -> tuple[float, float]:
        num_valid_tokens = 0
        num_total_tokens = 0
        num_valid_reqs = 0
        num_total_reqs = 0
        for r in req_results:
            num_total_tokens += r.prompt_len + r.output_len-1
            num_total_reqs += 1
            if check_is_prefill_latency_valid(r, ttft_tol, slo_lower_bound) and r.tpot*1000 <= tpot_slo:
                num_valid_tokens += r.prompt_len + r.output_len-1
                num_valid_reqs += 1
        if num_total_reqs == 0:
            return 1, 1
        return num_valid_tokens / num_total_tokens, num_valid_reqs / num_total_reqs

    def get_ttft_slo_attainment(req_results: list[ReqResult], ttft_tol: float, ttft_slo_lower_bound: float) -> tuple[float, float]:
        num_valid_tokens = 0
        num_total_tokens = 0
        num_valid_reqs = 0
        num_total_reqs = 0
        for r in req_results:
            num_total_tokens += r.prompt_len
            num_total_reqs += 1
            if check_is_prefill_latency_valid(r, ttft_tol, ttft_slo_lower_bound):
                num_valid_tokens += r.prompt_len
                num_valid_reqs += 1
        if num_total_reqs == 0:
            return 1, 1
        return num_valid_tokens / num_total_tokens, num_valid_reqs / num_total_reqs

    def get_tpot_slo_attainment(req_results: list[ReqResult], slo: float) -> tuple[float, float]:
        num_valid_tokens = 0
        num_total_tokens = 0
        num_valid_reqs = 0
        num_total_reqs = 0
        for r in req_results:
            if r.output_len <= 1:
                continue
            num_total_tokens += r.output_len-1
            num_total_reqs += 1
            if r.tpot*1000 <= slo:
                num_valid_tokens += r.output_len-1
                num_valid_reqs += 1
        if num_total_reqs == 0:
            return 1, 1
        return num_valid_tokens / num_total_tokens, num_valid_reqs / num_total_reqs
        
    ablation_ttft_tols = [
        25,
        25,
        25,
    ]
    ablation_ttft_slo_lower_bounds = [
        15000,
        15000,
        15000,
    ]
    ablation_tpot_slos = [
        50,
        50,
        50,
    ]
    def plot_ablation_slo_atta_with_rate():
        plt.rcParams.update({"font.size": 14})    
        def draw_one_plot(
            fig,
            ax: mpl_axes.Axes,
            plot_j: int,
            backends: list[BackendConfig],
            row_configs: PlotRowConfig,
            x_limits: list[Union[int, float]],
            retriever: Callable[[BenchmarkMetrics, list[ReqResult], int], float],
        ):
            cur_config = row_configs[plot_j]

            if plot_j == 0:
                ax.set_ylabel("SLO Attainment (%)", fontsize=14)
            ax.set_xlabel("Request Rate (req/s)")

            intersect_xs = []
            for cur_backend in backends:
                cur_xs = []
                cur_ys = []
                for (file_path, _, _, num_prompts, req_rate) in walk_around_exp_results(cur_config.dataset_name, cur_backend.name):
                    metrics, req_results = load_metric(file_path)
                    cur_xs.append(req_rate)
                    cur_ys.append(retriever(metrics, req_results, plot_j))
                ax.plot(cur_xs, cur_ys, label=cur_backend.label, marker=cur_backend.marker, color=cur_backend.color)
                intersect_x = get_intersection_point(cur_xs, cur_ys, 90)
                intersect_xs.append(intersect_x)
                ax.axvline(x=intersect_x, ymin=0, ymax=90/110, color=cur_backend.color, linestyle='--')
    
            if plot_j == 0:
                fig.legend(frameon=False, loc="upper center", ncol=2, prop={'size': 16}, bbox_to_anchor=(0.5, 1.2))

            ax.axhline(y=90, color='grey', linestyle='--', label="SLO")
            ax.set_xlim(0, x_limits[plot_j])
            ax.set_ylim(0, 110)
            set_xinjin_style(ax)
            
            assert backends[0].label == "LoongServe"
            for (backend_i, backend) in enumerate(backends):
                cur_intersect_xs = intersect_xs[backend_i]
                longserve_intersect_xs = intersect_xs[0]
                speedup = longserve_intersect_xs / cur_intersect_xs
                print(f"{backend.label.replace('LoongServe', 'LS')}: {cur_intersect_xs:.3f} ({speedup:.2f}x)")

        backends = [
            BackendConfig("longserve", "LoongServe", "C0", "o"),
            BackendConfig("longserve-fixsp-tp8", "LoongServe w/o ESP (TP=8)", "C1", "s"),
            BackendConfig("longserve-fixsp-sp4tp2", "LoongServe w/o ESP (TP=2, SP=4)", "C2", "D"),
            BackendConfig("longserve-fixsp-tp2dp4", "LoongServe w/o ESP (TP=2) x 4", "C3", "^"),
        ]
        row_configs = [
            PlotRowConfig("zipf1.0", "Zipf with a=1.0"),
            PlotRowConfig("zipf1.2", "Zipf with a=1.2"),
            PlotRowConfig("zipf1.4", "Zipf with a=1.4"),
        ]
        x_limits = [
            1.5,
            3.7,
            11
        ]
        retriever = lambda x, req_results, row: get_both_slo_attainment(req_results, ablation_ttft_tols[row], ablation_ttft_slo_lower_bounds[row], ablation_tpot_slos[row])[1]*100
        
        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        draw_one_plot(fig, axs[0], 0, backends, row_configs, x_limits, retriever)
        draw_one_plot(fig, axs[1], 1, backends, row_configs, x_limits, retriever)
        draw_one_plot(fig, axs[2], 2, backends, row_configs, x_limits, retriever)
        
        save_fig("fig12.png")

    plot_ablation_slo_atta_with_rate()

if fig_id == "10":
    draw_fig_10()
elif fig_id == "11":
    draw_fig_11()
elif fig_id == "12":
    draw_fig_12()
else:
    raise ValueError(f"Invalid fig_id: {fig_id}")
