import uuid
import numpy as np
from typing import List, Tuple
from loongserve.longserve_server.io_struct import Batch, Req
from loongserve.longserve_server.router.req_queue import ReqQueue
from loongserve.longserve_server.io_struct import ReqRunStatus

class Strategy:

    def ordering_reqs(self, batch_list: list[Batch]) -> List[Tuple[Req, Batch]]:
        raise not NotImplemented()

class Fcfs(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch_list: list[Batch]):
        req_batches = [(req, batch) for batch in batch_list for req in batch.reqs]
        return sorted(req_batches, key=lambda req_batches: req_batches[0].request_id, reverse=True)

class Sfj(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def ordering_reqs(self, batch_list: list[Batch]):
        req_batches = [(req, batch) for batch in batch_list for req in batch.reqs]
        return sorted(req_batches, key=lambda req_batches: req_batches[0].max_output_len - len(req_batches[0].output_ids), reverse=True)

class Hrnn(Strategy):

    def __init__(self) -> None:
        super().__init__()
    
    def ordering_reqs(self, batch_list: list[Batch]):
        req_batches = [(req, batch) for batch in batch_list for req in batch.reqs]
        return sorted(req_batches, key=lambda req_batches: (req_batches[0].input_len + req_batches[0].max_output_len - len(req_batches[0].output_ids)) / req_batches[0].input_len, reverse=True)


def select_paused_reqs(batch_list: list[Batch], strategy: Strategy, req_queue: ReqQueue, max_total_token_num):
    req_batches = strategy.ordering_reqs(batch_list)
    pause_req, pause_batch = req_batches[0]
    pause_batch.pop_req(pause_req.request_id)

    pause_req.req_status = ReqRunStatus.PAUSED_AND_OFFLOAD
    pause_req.cur_kv_len_list[:] = 0
    
    req_queue.back_to_wait_list([pause_req])

    return [(pause_req, pause_batch)]

