import csv
from typing import List, Tuple
import numpy as np
from loongserve.utils.log_utils import init_logger

logger = init_logger(__name__)

class Profiler:
    def __init__(
        self,
        profiler_file_path: str
    ):
        self.filename = profiler_file_path
        
        self.predictor_parameters = np.full((9, 9, 3), fill_value=float('nan'), dtype=np.float64)
        
        with open(self.filename, 'r') as f:
            reader = csv.DictReader(f)
            print(f"start initializing Profiler, file: {self.filename}", flush=True)
            for row in reader:
                sp_world_size = int(row['sp_world_size'])
                tp_world_size = int(row['tp_world_size'])
                
                A = float(row['A'])
                B = float(row['B'])
                C = float(row['C'])
                
                self.predictor_parameters[sp_world_size][tp_world_size] = (A, B, C)
    
    def predict_time_consumption_one_req(
        self,
        prompt_len: int,
        sp_world_size: int = 4,
        tp_world_size: int = 2
    ) -> float:
        A, B, C = self.predictor_parameters[sp_world_size][tp_world_size]
        assert not np.isnan(A) and not np.isnan(B) and not np.isnan(C)
        return A + B * prompt_len + C * prompt_len ** 2
    