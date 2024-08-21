"""
Start a proxy and (potential a lot of, if data parallel is enabled) API servers.
The proxy acts as a load balancer which uses round-robin to distribute the requests to the API servers.
"""
import os, sys
import time
import subprocess
import argparse
import multiprocessing

from lib.common import MODEL_PATH_LWM, MODEL_PATH_LWM_DISTSERVE, bcolors
from lib_benchmark_serving.backend_request_func import BACKEND_TO_PORTS

def api_server_starter_routine(
    port: int,
    worker_index: int,
    start_gpu_id: int,
    args: argparse.Namespace
):
    def get_vllm_max_params(args):
        max_model_len = \
            105000 if args.tp == 1 else \
            220000 if args.tp == 2 else \
            410000 if args.tp == 4 else \
            500000 if args.tp == 8 else \
            -1
        return max_model_len
    
    def get_lightllm_params(args):
        max_total_token_num = \
            110000 if args.tp == 1 else \
            210000 if args.tp == 2 else \
            500000 if args.tp == 4 else \
            900000 if args.tp == 8 else \
            -1
        running_max_req_size = {
            "sharegpt": 1024,
            "leval": 256,
            "lv-eval": 256,
            "mixed1": 1024,
            "mixed2": 1024,
            "synthesised": 1024,
            "zipf1": 1024,
            "sched-zipf": 1536,
        }[args.dataset]
        max_num_ooe = {
            "sharegpt": 1,
            "leval": 8,
            "lv-eval": 8,
            "mixed1": 4,
            "mixed2": -1,
            "synthesised": 1,
            "zipf1": 64,
            "sched-zipf": 64
        }[args.dataset]
        return max_total_token_num, running_max_req_size, max_num_ooe
    
    """
    Start the target API server on the target port
    """
    assert args.tp in [1, 2, 4, 8]
    gpus_per_worker = args.tp * args.pp * args.sp
    total_gpus = gpus_per_worker * args.dp
    gpu_ids = ",".join([
        str(start_gpu_id + i)
        for i in range(gpus_per_worker)
    ])
    print(f"Worker starting with {port=}")


    if args.backend == "vllm":
        assert args.sp == 1, "Sequence parallelism is not supported in VLLM."
        max_model_len = get_vllm_max_params(args)
        # NOTE Here we do not set CUDA_VISIBLE_DEVICES since ray will set it automatically
        script = f"""
cd /mnt/petrelfs/zhaoyihao/intlsy/research;
. vllm/start-env.fish;
python -u -m vllm.entrypoints.api_server \\
    --host 0.0.0.0 --port {port} \\
    --engine-use-ray --disable-log-requests \\
    --model {args.model} --dtype float16 --worker-use-ray \\
    -pp {args.pp} -tp {args.tp} \\
    --block-size 16 --seed 0 \\
    --enforce-eager --disable-custom-all-reduce \\
    --max-model-len {max_model_len} --gpu-memory-utilization 0.98 \\
    --swap-space 32 \\
    --tokenizer-mode auto \\
    --max-num-batched-tokens 600000 \\
    --max-num-seqs 1024
        """

        
    elif args.backend == "lightllm-sf":
        assert args.pp == 1, "Pipeline parallelism is not supported in LightLLM."
        assert args.sp == 1, "Sequence parallelism is not supported in LightLLM."
        max_total_token_num, running_max_req_size, _ = get_lightllm_params(args)
        max_req_len = min(max_total_token_num, 500000)
        splitfuse_block_size = {
            "sharegpt": 256,
            "leval": 2048,
            "lv-eval": 16384,
            "mixed1": 2048,
            "mixed2": -1,
            "zipf1": 512
        }[args.dataset]
        script = f"""
cd /mnt/petrelfs/zhaoyihao/intlsy/research;
. lightllm-sf/start-env.fish;
python -u -m lightllm.server.api_server \\
    --host 0.0.0.0 --port {port} \\
    --model_dir {args.model} --tokenizer_mode auto \\
    --max_total_token_num {max_total_token_num} --running_max_req_size {running_max_req_size} \\
    --tp {args.tp} \\
    --max_req_input_len {max_req_len-1} --max_req_total_len {max_req_len} \\
    --mode triton_flashdecoding \\
    --batch_max_tokens 600000 \\
    --splitfuse_mode --splitfuse_block_size {splitfuse_block_size}
        """
        

    elif args.backend == "longserve" or args.backend == "longserve-fixsp":
        assert args.pp == 1, "Pipeline parallelism is not supported in LongServe."
        max_total_token_num, running_max_req_size, max_num_ooe = get_lightllm_params(args)
        max_req_len = min(max_total_token_num * args.sp, 500000)
        script = f"""
export CUDA_VISIBLE_DEVICES={gpu_ids};
cd /mnt/petrelfs/zhaoyihao/intlsy/research/LongServe;
python -u -m loongserve.longserve_server.api_server \\
    --host 0.0.0.0 --port {port} \\
    --model_dir {args.model} --tokenizer_mode auto \\
    --max_total_token_num {max_total_token_num} --running_max_req_size {running_max_req_size} \\
    --tp_world_size {args.tp} --sp_world_size {args.sp} \\
    --max_req_input_len {max_req_len-1} --max_req_total_len {max_req_len} \\
    --mode _token_decode_attention_overlapped \\
    --batch_max_tokens 500000 \\
    --max_mig_len 10000 \\
    --avg_decoding_time {22 if args.tp*args.sp <= 8 else 25} \\
    --nccl_port {28768+worker_index} \\
    --log_stats_interval 5 \\
    --max_prefill_time 5000 \\
    --local_world_size {min(gpus_per_worker, 8)} \\
    --max_wait_tokens 10 \\
    --min_comp_bound_decoding_batch_size 128 \\
    --profiler_file_path /mnt/petrelfs/zhaoyihao/intlsy/research/exp-results/{args.ae_id}/analytical-model.csv \\
    --max_num_ooe {max_num_ooe} {"--use_fixed_sp" if args.backend == "longserve-fixsp" else ""} \\
    {f"--disable_scale_up" if args.disable_scale_up else ""} \\
    {f"--with_log_trace {args.with_log_trace}" if args.with_log_trace else ""}
""" # FIXME: remove the hard-coded path
    

    elif args.backend == "deepspeed":
        assert args.pp == 1, "Pipeline parallelism is not supported in DeepSpeed."
        assert args.sp == 1, "Pipeline parallelism is not supported in DeepSpeed."
        script = f"""
cd /mnt/petrelfs/zhaoyihao/intlsy/research;
. DeepSpeed-MII/start-env.fish;
unset https_proxy; unset http_proxy; unset all_proxy;
unset HTTPS_PROXY; unset HTTP_PROXY; unset ALL_PROXY;
python -m mii.entrypoints.api_server \\
    --model {args.model} \\
    --port {port} \\
    --host 0.0.0.0 \\
    --max-length 500000 \\
    --tensor-parallel {args.tp}
"""
    

    elif args.backend == "distserve":
        if args.model == MODEL_PATH_LWM:
            args.model = MODEL_PATH_LWM_DISTSERVE
        script = f"""
cd /mnt/petrelfs/zhaoyihao/intlsy/research;
. DistServe/start-env.fish;
python DistServe/distserve/api_server/distserve_api_server.py \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer hf-internal-testing/llama-tokenizer \\
    --use-dummy-weights \\
    \\
    --context-pipeline-parallel-size 1 \\
    --context-tensor-parallel-size 4 \\
    --decoding-pipeline-parallel-size 2 \\
    --decoding-tensor-parallel-size 2 \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 37500 \\
    --gpu-memory-utilization 0.82 \\
    --swap-space 32 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 550000 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 128 \\
    --decoding-max-tokens-per-batch 550000
"""

    elif args.backend == "dummy":
        script = f"""
echo worker with {start_gpu_id=} starts;
python3 -m http.server {port}
"""
    
    print(f"Starting server with command {script}")
    subprocess.run(["bash", "-c", script]) # FIXME: remove the hard-coded shell name

def proxy_routine(
    port: int,
    worker_ports: list[int],
    args: argparse.Namespace
):
    """
    Start a proxy server, which listens to `port` and forward TCP connections
    to `worker_ports`, in a round_robin manner
    """
    
    import socket, threading, select
    def proxy_worker_thread_routine(
        client_sock: socket.socket,
        worker_port: int
    ):
        print(f"New request accepted, forwarding to worker on port {worker_port}")
        # Connect to worker
        worker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        worker_sock.connect(("localhost", worker_port))
        # Forward data between client and worker
        while True:
            r, _, _ = select.select([client_sock, worker_sock], [], [])
            if client_sock in r:
                data = client_sock.recv(4096)
                if not data:
                    break
                worker_sock.sendall(data)
            if worker_sock in r:
                data = worker_sock.recv(4096)
                if not data:
                    break
                client_sock.sendall(data)
        
    next_worker_id = 0  # The next worker to forward the request to
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", port))
    sock.listen(128)
    print(f"The proxy server is listening on port {port}")
    while True:
        client_sock, addr = sock.accept()
        cur_thread = threading.Thread(
            target = proxy_worker_thread_routine,
            args = (client_sock, worker_ports[next_worker_id])
        )
        cur_thread.start()
        next_worker_id = (next_worker_id + 1) % len(worker_ports)


def main(args: argparse.Namespace):
    print(args)
    gpus_per_worker = args.tp * args.pp * args.sp
    assert os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7") == "0,1,2,3,4,5,6,7", "We highly recommend you to run the servers on a exclusive node"
    proxy_port = BACKEND_TO_PORTS[args.backend]
    
    if args.dp == 1:
        # We only have 1 DP worker. Launch the api server on the current process
        # and set port to proxy_port
        api_server_starter_routine(
            proxy_port,
            0,
            0,
            args
        )
        sys.exit(0)

    processes = []    
    for worker_id in range(args.dp):
        # Start workers on port [proxy_port + 1, proxy_port + args.dp]
        process = multiprocessing.Process(
            target = api_server_starter_routine,
            args = (proxy_port + worker_id + 1, worker_id, worker_id * gpus_per_worker, args)
        )
        processes.append(process)
        process.start()
        time.sleep(60)
    
    # Start the proxy server
    process = multiprocessing.Process(
        target = proxy_routine,
        args = (proxy_port, list(range(proxy_port + 1, proxy_port + args.dp + 1)), args)
    )
    processes.append(process)
    process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help="The serving backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH_LWM,
        help="The model to be served"
    )
    parser.add_argument(
        "-tp",
        type=int,
        required=True,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "-pp",
        type=int,
        default=1,
        help="Pipeline parallel size"
    )
    parser.add_argument(
        "-dp",
        type=int,
        default=1,
        help="Data parallel size"
    )
    parser.add_argument(
        "-sp",
        type=int,
        default=1,
        help="Sequence parallel size"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The target dataset",
        choices=["sharegpt", "leval", "lv-eval", "mixed1", "mixed2", "synthesised", "zipf1", "sched-zipf"]
    )
    parser.add_argument(
        "--with-log-trace",
        type=str,
        default=None,
        help="If provided with a path to a file, the trace will be logged to the file"
    )
    parser.add_argument(
        "--ae-id",
        type=str,
        default="intlsy"
    )
    parser.add_argument("--disable_scale_up", action='store_true', default=False)

    args = parser.parse_args()
    main(args)
