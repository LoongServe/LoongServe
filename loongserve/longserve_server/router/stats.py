import time
import json
from loongserve.utils.log_utils import init_logger
from ..io_struct import Batch

logger = init_logger(__name__)

class Stats:

    def __init__(self, log_status, with_log_trace, log_stats_interval) -> None:
        self.log_stats = log_status
        self.with_log_trace = with_log_trace
        self.log_stats_interval = log_stats_interval
        self.last_log_time = time.time()
        self.all_tokens = 0
        self.output_tokens = 0
        self.prompt_tokens = 0
        self.prompt_batch_size_sum = 0
        self.prompt_batch_count = 0
        self.decoding_batch_size_sum = 0
        self.decoding_batch_count = 0
        self.last_num_finished_reqs = 0
        if self.with_log_trace is not None:
            logger.debug(f"Trace logging is enabled. Trace will be written to {self.with_log_trace}")
            self.trace_file = open(self.with_log_trace, "w")
            self.trace_file.write("[\n")
        return
    
    def on_prompt_batch_start(self, sp_rank: int, run_batch: Batch):
        if self.with_log_trace is not None:
            trace_data = {
                "name": f"{len(run_batch.reqs)}",
                "cat": "Prefill",
                "pid": sp_rank,
                "ph": "B",
                "ts": time.perf_counter()*1e6,
                "cname": "cq_build_passed"
            }
            self.trace_file.write(f"{json.dumps(trace_data)},\n")

    def on_prompt_batch_finish(self, sp_rank: int,run_batch: Batch):
        if self.log_stats:
            tokens = run_batch.input_tokens()
            self.prompt_tokens += tokens
            self.all_tokens += tokens
            self.prompt_batch_size_sum += len(run_batch.reqs)
            self.prompt_batch_count += 1
        if self.with_log_trace is not None:
            trace_data = {
                "name": f"{len(run_batch.reqs)}",
                "cat": "Prefill",
                "pid": sp_rank,
                "ph": "E",
                "ts": time.perf_counter()*1e6,
                "cname": "cq_build_passed"
            }
            self.trace_file.write(f"{json.dumps(trace_data)},\n")
    
    def on_decoding_batch_start(self, sp_rank: int, run_batch: Batch):
        if self.with_log_trace is not None:
            trace_data = {
                "name": f"{len(run_batch.reqs)}",
                "cat": "Decode",
                "pid": sp_rank,
                "ph": "B",
                "ts": time.perf_counter()*1e6,
                "cname": "cq_build_failed"
            }
            self.trace_file.write(f"{json.dumps(trace_data)},\n")

    def on_decoding_batch_finish(self, sp_rank: int, run_batch: Batch):
        if self.log_stats:
            tokens = len(run_batch.reqs)
            self.output_tokens += tokens
            self.all_tokens += tokens
            self.decoding_batch_size_sum += len(run_batch.reqs)
            self.decoding_batch_count += 1
        if self.with_log_trace is not None:
            trace_data = {
                "name": f"{len(run_batch.reqs)}",
                "cat": "Decode",
                "pid": sp_rank,
                "ph": "E",
                "ts": time.perf_counter()*1e6,
                "cname": "cq_build_failed"
            }
            self.trace_file.write(f"{json.dumps(trace_data)},\n")

    def print_stats(self, cur_log_time, num_finished_reqs):
        if not self.log_stats:
            return

        log_duration = cur_log_time - self.last_log_time
        if log_duration > self.log_stats_interval:
            logger.debug(f"Avg request throughput:  {(num_finished_reqs-self.last_num_finished_reqs) / log_duration:8.3f} reqs/s\n"
                         f"Avg overall throughput:  {self.all_tokens / log_duration:8.3f} tokens/s\n"
                         f"Avg prompt throughput:   {self.prompt_tokens / log_duration:8.3f} tokens/s\n"
                         f"Avg decoding throughput: {self.output_tokens / log_duration:8.3f} tokens/s\n"
                         f"Avg prompt batch size:   {(self.prompt_batch_size_sum/self.prompt_batch_count if self.prompt_batch_count != 0 else 0):8.3f}\n"
                         f"Avg decoding batch size: {(self.decoding_batch_size_sum/self.decoding_batch_count if self.decoding_batch_count != 0 else 0):8.3f}"
                        )
            self.last_num_finished_reqs = num_finished_reqs
            self.all_tokens = 0
            self.output_tokens = 0
            self.prompt_tokens = 0
            self.prompt_batch_size_sum = 0
            self.prompt_batch_count = 0
            self.decoding_batch_size_sum = 0
            self.decoding_batch_count = 0
            self.last_log_time = cur_log_time
            if self.with_log_trace is not None:
                self.trace_file.flush()
        return

    