import torch
from loongserve.utils.log_utils import init_logger
from loongserve.common.longserve_mem_manager import LongServeMemoryManager

logger = init_logger(__name__)
    
class LongServeReqManager:
    """
    ReqManager - Manage the mapping between request index and token slot index
    
    In TokenAttention, the K/V cache of a particular request may be allocated
    non-contiguously in the huge K/V cache array, which forces us to maintain
    a mapping between the request index and the token slot index. ReqManager is
    used to manage this mapping.
    """
    
    def __init__(self, max_request_num, max_sequence_length, mem_manager):
        self.req_state = torch.zeros((max_request_num,), dtype=torch.bool, device="cuda")
        self.req_to_token_indexs = torch.full((max_request_num, max_sequence_length), fill_value=2147483647, dtype=torch.int32, device="cuda")
        self.can_use_req_size = max_request_num
        self.mem_manager: LongServeMemoryManager = mem_manager

    def alloc(self, need_size):
        if need_size > self.can_use_req_size:
            logger.error(f'Insufficient requested capacity, remaining {self.can_use_req_size}')
            return None
        select_index = torch.nonzero(self.req_state==0).reshape(-1)[:need_size]
        self.req_state[select_index] = 1
        self.can_use_req_size -= len(select_index)
        return select_index
    
    def free(self, free_req_index, free_token_index):
        self.can_use_req_size += len(free_req_index)
        self.req_state[free_req_index] = 0
        if free_token_index is not None:
            self.mem_manager.free(free_token_index)
    
    def free_req(self, free_req_index):
        self.can_use_req_size +=1
        self.req_state[free_req_index] = 0
        return
    
    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)

    def free_all(self):
        self.can_use_req_size = len(self.req_state)
        self.req_state[:] = 0
    