import torch
import vllm
import ray

from lib.structs import *
from lib.sut import SystemUnderTest, get_input_ids

class VLLMSUT(SystemUnderTest):
    def __init__(
        self,
        worker_param: WorkerParam
    ):
        assert worker_param.sp_world_size == 1

        self.worker_param = worker_param
        self.engine = vllm.LLM(
            model = worker_param.model_dir,
            dtype = "float16",
            kv_cache_dtype = "auto",
            max_model_len = worker_param.max_seq_len,
            worker_use_ray = True,
            pipeline_parallel_size = 1,
            tensor_parallel_size = worker_param.tp_world_size,
            max_num_seqs = worker_param.max_req_num,
            max_context_len_to_capture = worker_param.max_seq_len,
            enforce_eager = True,
            gpu_memory_utilization=0.995,
            disable_custom_all_reduce = True
        )

    def inference(
        self,
        input_param: InputParam
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], float, list[float]]:
        input_ids = get_input_ids(self.worker_param.model_dir, input_param.input_len*input_param.batch_size)
        prompt_token_ids = input_ids.view(input_param.batch_size, input_param.input_len).tolist()
        sampling_params = vllm.sampling_params.SamplingParams(
            ignore_eos = True,
            max_tokens = input_param.output_len,
        )
        request_outputs, prefill_time_usage, decoding_time_usages = self.engine.generate(
            prompt_token_ids = prompt_token_ids,
            sampling_params = sampling_params,
            use_tqdm = False,
            timing = True
        )
        predict_ids = [request_output.outputs[0].token_ids for request_output in request_outputs]
        predict_texts = [request_output.outputs[0].text for request_output in request_outputs]
        return input_ids, predict_ids, predict_texts, prefill_time_usage, decoding_time_usages
    