import csv
from typing import List, Tuple
from ..io_struct import Batch, Req
import numpy as np
from loongserve.utils.log_utils import init_logger

logger = init_logger(__name__)

class Profiler:
    def __init__(self, args):
        self.filename = args.profiler_file_path
        
        self.total_world_size = args.total_world_size
        self.tp_world_size = args.tp_world_size
        self.sp_world_size = args.sp_world_size
        assert self.total_world_size == self.tp_world_size * self.sp_world_size
        
        self.predictor_parameters = np.full((self.sp_world_size + 1, 3), fill_value=float('nan'), dtype=np.float64)
        
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            print(f"start initializing Profiler, file: {self.filename}", flush=True)
            for row in reader:
                sp_world_size = int(row['sp_world_size'])
                tp_world_size = int(row['tp_world_size'])
                
                if sp_world_size > self.sp_world_size or tp_world_size != self.tp_world_size:
                    continue
                
                logger.debug(row)
                
                A = float(row['A'])
                B = float(row['B'])
                C = float(row['C'])
                
                self.predictor_parameters[sp_world_size] = (A, B, C)
            
            assert np.all(np.isnan(self.predictor_parameters[1:]) == False)
    
    def predict(self, sp_world_size:int, req_input_sum: int, req_input_square_sum: int) -> float:
        assert sp_world_size <= self.sp_world_size
        
        A, B, C = self.predictor_parameters[sp_world_size]
        return A + B * req_input_sum + C * req_input_square_sum