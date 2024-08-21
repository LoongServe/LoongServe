import dataclasses

"""
We divide parameters into two types:
- Worker parameters, including model_dir, sp_world_size, tp_world_size, and "mode". When these parameter
  change, we need to re-create workers
- Input parameters, including batch_size, input_len, output_len, seq_block_size.
  We do not need to re-create workers when these parameters change
"""

@dataclasses.dataclass
class WorkerParam:
	"""
	Worker parameters, defining how decoding is performed
	"""
	model_dir: str
	sp_world_size: int
	tp_world_size: int
	max_total_token_num: int
	max_req_num: int
	max_seq_len: int
	mode: list[str]

@dataclasses.dataclass
class InputParam:
	"""
	Input parameters, defining the request data
	"""
	batch_size: int
	input_len: int
	output_len: int
	num_sp_master: int
	need_context_migration: bool
	num_decoding_stage_migration: int

@dataclasses.dataclass
class TestParamGroup:
	"""
	One WorkerParam coupled with multiple InputParam
	"""
	worker_param: WorkerParam
	input_params: list[InputParam]