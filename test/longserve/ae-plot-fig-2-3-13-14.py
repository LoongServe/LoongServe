import os, sys, math
from typing import Optional, Callable
import argparse
import csv

from lib.structs import WorkerParam, InputParam, TestParamGroup
from lib.db import RecordManager
from lib.sut import SystemUnderTest
from lib.run_test_param_group import run_test_params

parser = argparse.ArgumentParser()
parser.add_argument("--ae-id", type=str, default="intlsy")
parser.add_argument("--fig", type=str, required=True)
args = parser.parse_args()
ae_id = args.ae_id
fig_id = args.fig

model_dir = os.environ.get("LWM_WEIGHT_PATH", "Env `LWM_WEIGHT_PATH` is not set!")
db_dir = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}/loongserve-db-identical-req.sqlite"
figs_dir = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}"
analytical_model_path = f"/mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{ae_id}/analytical-model.csv"

record_manager = RecordManager(filename=db_dir)

def save_fig(filename: str):
    if not filename.endswith(".png"):
        filename += ".png"
    plt.savefig(os.path.join(figs_dir, filename), bbox_inches='tight')

def _fetch_result(
    worker_param: WorkerParam,
    input_param: InputParam,
    is_not_found_ok: bool
) -> Optional[tuple[float, float]]:
    """
    Fetch the experiment result (prefill_time, decoding_time) from the db
    """
    cached_result = record_manager.query_record(worker_param, input_param)
    if cached_result is not None:
        return cached_result["avg_prefill_time_usage"], cached_result["avg_decoding_time_usage"]
    if not is_not_found_ok:
        raise ValueError(f"Result for {worker_param} {input_param} not found!")
    else:
        return None

def fetch_result(
    sp_world_size: int,
    tp_world_size: int,
    batch_size: int,
    input_len: int,
    output_len: int,
    num_sp_master: int = 1,
    need_context_migration: bool = False,
    num_decoding_stage_migration: int = 0,
    is_not_found_ok: bool = False
) -> Optional[tuple[float, float]]:
    """
    A wrapper around `_fetch_result_or_run` that provides a simplified interface.
    """
    worker_param = WorkerParam(
        model_dir,
        sp_world_size,
        tp_world_size,
        0,
        0,
        0,
        ["_token_decode_attention_overlapped"]
    )
    input_param = InputParam(
        batch_size,
        input_len,
        output_len,
        num_sp_master,
        need_context_migration,
        num_decoding_stage_migration
    )
    return _fetch_result(worker_param, input_param, is_not_found_ok)

from matplotlib import pyplot as plt
import matplotlib.axes as mpl_axes
import numpy as np
from matplotlib import ticker as mticker

def set_xinjin_style(ax: mpl_axes.Axes):
    ax.tick_params('x', direction="in")
    ax.tick_params('y', direction="in")
    ax.spines['top'].set_color(None)
    ax.spines['right'].set_color(None)
     
def draw_fig_2():
    plt.rcParams.update({"font.size": 16})
    tp_world_sizes = [1, 2, 4, 8]
    batch_size_and_input_lens_short_requests = [
        (16, 10, "10"),
        (16, 50, "50"),
        (16, 100, "100"),
        (16, 500, "500"),
    ]
    batch_size_and_input_lens_long_requests = [
        (1, 100, "100"),
        (1, 1000, "1k"),
        (1, 10000, "10k"),
        (1, 100000, "100k"),
    ]
    markers = ['o', 's', 'D', '^']
    output_len = 16

    def plot_figure(ax: mpl_axes.Axes, title: str, batch_size_and_input_lens: list, is_decoding_stage: bool, normalize: bool):
        retriever = lambda x: x[1] if is_decoding_stage else x[0]
        data = np.array([
            [
                retriever(fetch_result(
                    1,
                    tp_world_size,
                    batch_size,
                    input_len,
                    output_len
                )) if batch_size*input_len <= 100000*tp_world_size else 0
                for (batch_size, input_len, _) in batch_size_and_input_lens
            ]
            for tp_world_size in tp_world_sizes
        ])	# [num_tps, num_batch_size_input_lens, 2]
        for i in range(len(batch_size_and_input_lens)):
            cur_data = data[:, i]
            if normalize:
                cur_data /= np.max(cur_data)
            ax.plot(
                tp_world_sizes,
                cur_data, 
                marker=markers[i],
                label=f"BS={batch_size_and_input_lens[i][0]}, Len={batch_size_and_input_lens[i][2]}", linewidth=1
            )
        
        linear_scaling_xs = np.arange(1, 8, 0.02)
        ax.plot(
            linear_scaling_xs,
            [1/x for x in linear_scaling_xs],
            "--",
            label="Linear Scaling",
            color="grey"
        )

        # ax.set_title(title)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position((box.x0, box.y0, box.width, box.height*0.78))
        ax.set_xlabel("Degree of Tensor Parallelism (TP)")
        set_xinjin_style(ax)
        if normalize:
            ax.set_ylim(0, 1.05)
            
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0][0].set_ylabel("Normalized Iteration Time (s)")
    axs[1][0].set_ylabel("Normalized Iteration Time (s)")

    plot_figure(axs[0][0], "Prefill Phase, Short Request", batch_size_and_input_lens_short_requests, is_decoding_stage=False, normalize=True)
    plot_figure(axs[0][1], "Decoding Phase, Short Request", batch_size_and_input_lens_short_requests, is_decoding_stage=True, normalize=True)
    plot_figure(axs[1][0], "Prefill Phase, Long Request", batch_size_and_input_lens_long_requests, is_decoding_stage=False, normalize=True)
    plot_figure(axs[1][1], "Decoding Phase, Long Request", batch_size_and_input_lens_long_requests, is_decoding_stage=True, normalize=True)

    axs[0][0].legend(frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(1.1, 1.3))
    axs[1][0].legend(frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(1.1, 1.3))

    save_fig("fig2.png")
    plt.show()

def draw_fig_3():
    plt.rcParams.update({"font.size": 14})
    sp_and_tp_world_sizes = [
        (1, 8),
        (2, 4),
        (4, 2),
    ]
    batch_size_and_input_lens = [
        (512, 1000, [1, 2, 4]),
        (128, 5000, [1, 1, 2]),
        (64, 10000, [1, 1, 2]),
        (16, 50000, [1, 1, 1]),
        (4, 100000, [1, 1, 1]),
        (1, 500000, [1, 1, 1])
    ]
    output_len = 16
    bar_width = 0.25

    data = np.array([
        [ 
            fetch_result(
                sp_world_size,
                tp_world_size,
                batch_size,
                input_len,
                output_len,
                num_sp_master = num_sp_masters[sp_tp_index]
            )
            for (batch_size, input_len, num_sp_masters) in batch_size_and_input_lens
        ]
        for sp_tp_index, (sp_world_size, tp_world_size) in enumerate(sp_and_tp_world_sizes)
    ])	# [num_tps, num_batch_size_input_lens, 2]
    # Normalization (respective to SP=1)
    print(data)
    assert sp_and_tp_world_sizes[0][0] == 1
    data /= data[0, :, :]

    def plot_figure(ax: mpl_axes.Axes, title: str, retriver: Callable):
        xs_base = np.arange(len(batch_size_and_input_lens))
        for i, (sp_world_size, tp_world_size) in enumerate(sp_and_tp_world_sizes):
            xs = xs_base + bar_width*i
            label = f"SP={sp_world_size}, TP={tp_world_size}"
            cur_data = [retriver(x) for x in data[i]]
            ax.bar(xs, cur_data, bar_width, label=label, edgecolor="black")
            
        xticks = []
        for (i, (batch_size, input_len, _)) in enumerate(batch_size_and_input_lens):
            input_len_text = (str(input_len // 1000) + "k") if input_len >= 1000 else input_len
            if i == 0:
                xticks.append(f"{batch_size}\n{input_len_text}")
            else:
                xticks.append(f"{batch_size}\n{input_len_text}")
        ax.set_xticks(xs_base + bar_width*(len(sp_and_tp_world_sizes)-1)/2, xticks)
        ax.axhline(y = 1.0, color="grey", linestyle="--")
        ax.text(-0.8, -0.022, " BS=\nLen=", ha='center', va='top', color='black', fontsize=12)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        set_xinjin_style(ax)
            
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_ylabel("Normalized Iteration Time (s)")
    
    plot_figure(ax0, "Prefill Phase", lambda x: x[0])
    ax0.set_ylim(0, 1.2)
    ax1.set_ylim(0, 1.2)
    fig.legend(frameon=False, loc="upper center", ncol=4, prop={'size': 16}, bbox_to_anchor=(0.5, 1.02))
    plot_figure(ax1, "Decoding Phase", lambda x: x[1])

    save_fig("fig3.png")
    plt.show()


def draw_fig_13():
    def get_input_len_text(input_len: int) -> str:
        if input_len < 1000:
            return f"{input_len}"
        else:
            return f"{input_len//1000}K"
	
    batch_size_and_input_lens = [
        (1024, 10), (256, 100), (64, 1000), (16, 10000), (4, 50000), (2, 100000), (1, 200000)
    ]
    def draw_scale_down_plot(ax: mpl_axes.Axes):
        plt.rcParams.update({"font.size": 14})
        xs_base = np.arange(len(batch_size_and_input_lens))
        bar_width = 0.38
        
        data = np.array([
            [
                fetch_result(4, 2, batch_size, input_len, 16, 1, perform_context_migration, 0)[0]
                for (batch_size, input_len) in batch_size_and_input_lens
            ]
            for perform_context_migration in [False, True]
        ])
        data /= data[0]

        for perform_context_migration in [False, True]:
            xs = xs_base + bar_width*(1 if perform_context_migration else 0)
            label = "Prefill w/ Scale Down" if perform_context_migration else "Prefill w/o Scale Down"
            cur_data = data[1 if perform_context_migration else 0]
            ax.bar(xs, cur_data, bar_width, label=label, edgecolor="black")

        xticks = [f"{batch_size}\n{get_input_len_text(input_len)}" for (batch_size, input_len) in batch_size_and_input_lens]
        ax.set_xticks(xs_base + bar_width/2, xticks, size=13)
        set_xinjin_style(ax)
        ax.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.25))
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(-0.9, -0.02, " BS=\nLen=", ha='center', va='top', color='black', fontsize=12)

    def draw_scale_up_plot(ax: mpl_axes.Axes):
        plt.rcParams.update({"font.size": 14})
        num_sp_masters = [1, 2, 4]
        xs_base = np.arange(len(batch_size_and_input_lens))
        bar_width = 0.25
        
        data = np.array([
            [
                fetch_result(4, 2, batch_size, input_len, 16, num_sp_master, False, 0)[1]
                for (batch_size, input_len) in batch_size_and_input_lens
            ]
            for num_sp_master in num_sp_masters
        ])
        data /= data[0]

        for (index, num_sp_master) in enumerate(num_sp_masters):
            xs = xs_base + bar_width*(index+0.5)
            label = f"Decoding w/ {num_sp_master} SP Master{'s' if num_sp_master > 1 else ''}"
            cur_data = data[index]
            ax.bar(xs, cur_data, bar_width, label=label, edgecolor="black")

        xticks = [f"{batch_size}\n{get_input_len_text(input_len)}" for (batch_size, input_len) in batch_size_and_input_lens]
        ax.set_xticks(xs_base + bar_width*len(num_sp_masters)/2, xticks, size=13)
        set_xinjin_style(ax)
        ax.legend(frameon=False, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.3))
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(-0.8, -0.02, " BS=\nLen=", ha='center', va='top', color='black', fontsize=12)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    ax0.set_ylabel("Normalized Latency (s)")
    draw_scale_down_plot(ax0)
    draw_scale_up_plot(ax1)
    save_fig("fig13.png")
    plt.show()

def draw_fig_14():
    class Profiler:
        def __init__(self):
            self.predictor_parameters: dict[tuple[int, int], tuple[float, float, float]] = {}
            with open(analytical_model_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sp_world_size = int(row['sp_world_size'])
                    tp_world_size = int(row['tp_world_size'])
                    A = float(row['A'])
                    B = float(row['B'])
                    C = float(row['C'])
                    self.predictor_parameters[(sp_world_size, tp_world_size)] = (A, B, C)
        
        def _predict(self, sp_world_size:int, tp_world_size:int, req_input_sum: int, req_input_square_sum: int) -> float:
            A, B, C = self.predictor_parameters[(sp_world_size, tp_world_size)]
            return A + B * req_input_sum + C * req_input_square_sum

        def predict(self, sp_world_size: int, tp_world_size: int, batch_size: int, input_len: int) -> float:
            return self._predict(sp_world_size, tp_world_size, batch_size * input_len, batch_size * input_len * input_len)  
    plt.rcParams.update({"font.size": 16})
    sp_and_tp_sizes = [
        (2, 4),
        (4, 2),
        (8, 1)
    ]
    batch_sizes = [1, 2 ,4, 8]
    colors = ["C0", "C1", "C2", "C3", "cyan", "purple", "brown", "yellow"]
    input_lens_pred = [
        10, 100, 1000
    ] + list(range(5000, 500000+1, 5000))
    input_lens_real = [
        10, 100, 1000, 10000, 20000, 50000, 100000, 200000, 400000, 500000
    ]
    output_len = 16
    markers = ['X', '+', 'x', '1']

    profiler = Profiler()
    def plot(ax: mpl_axes.Axes, sp_world_size: int, tp_world_size: int):
        for i, batch_size in enumerate(batch_sizes):
            real_data = []
            for input_len in input_lens_real:
                result = fetch_result(
                    sp_world_size,
                    tp_world_size,
                    batch_size,
                    input_len,
                    output_len,
                    is_not_found_ok = True
                )
                if result is None:
                    break
                else:
                    real_data.append(result[0]/1000)
            cur_input_lens_real = input_lens_real[:len(real_data)]

            cur_input_lens_pred = [input_len for input_len in input_lens_pred if input_len <= cur_input_lens_real[-1]]
            pred_data = np.array([
                profiler.predict(sp_world_size, tp_world_size, batch_size, input_len)/1000
                for input_len in cur_input_lens_pred
            ])
            ax.plot(cur_input_lens_pred, pred_data, color=colors[i], label=f"BS={batch_size} (Pred)", linewidth=1)
            ax.scatter(cur_input_lens_real, real_data, color=colors[i], marker=markers[i], s=70, label=f"BS={batch_size} (Real)")
            set_xinjin_style(ax)
            ax.set_xlabel("Input Length")

            ax.set_xticks([0, 100000, 200000, 300000, 400000, 500000])
            ax.set_xticklabels([f"{x/1000:.0f}k" for x in ax.get_xticks()])
            ax.set_ylim(0, 75)
            
    fig, axs = plt.subplots(1, len(sp_and_tp_sizes), figsize=(20, 4))
    for i in range(len(sp_and_tp_sizes)):
        plot(axs[i], sp_and_tp_sizes[i][0], sp_and_tp_sizes[i][1])
        if i == 0:
            fig.legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.11))
    axs[0].set_ylabel("Iteration Time (s)")

    save_fig("fig14.png")
    plt.show()


if fig_id == "2":
    draw_fig_2()
elif fig_id == "3":
    draw_fig_3()
elif fig_id == "13":
    draw_fig_13()
elif fig_id == "14":
    draw_fig_14()
else:
    raise ValueError(f"Invalid fig_id: {fig_id}")
