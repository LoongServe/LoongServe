import dataclasses
import marshal

@dataclasses.dataclass
class TestRequest:
    """
    TestRequest: A request for testing the server's performance
    """
    
    prompt: str
    prompt_len: int
    output_len: int
    
@dataclasses.dataclass
class Dataset:
    """
    Dataset: A dataset for testing the server's performance
    """
 
    dataset_name: str	# "sharegpt" / "alpaca" / ...
    data: list[TestRequest]
    
    def dump(self, output_path: str):
        marshal.dump({
            "dataset_name": self.dataset_name,
            "data": [(req.prompt, req.prompt_len, req.output_len) for req in self.data]
        }, open(output_path, "wb"))
    
    @staticmethod
    def load(input_path: str):
        loaded_data = marshal.load(open(input_path, "rb"))
        return Dataset(
            loaded_data["dataset_name"],
            [TestRequest(req[0], req[1], req[2]) for req in loaded_data["data"]]
        )
        
@dataclasses.dataclass
class ReqResult:
    prompt_len: int
    output_len: int

    issue_time: float
    first_token_time: float
    complete_time: float
    
    ttft: float
    tpot: float
    latency: float
    
    @staticmethod
    def from_http_request_result(
        prompt_len: int,
        output_len: int,
        issue_time: float,
        first_token_time: float,
        complete_time: float
    ):
        """
        benchmark-serving will call this function once a request is completed
        """
        return ReqResult(
            prompt_len,
            output_len,
            issue_time,
            first_token_time,
            complete_time,
            first_token_time - issue_time,
            (complete_time - first_token_time) / (output_len-1) if output_len > 1 else 0,
            complete_time - issue_time
        )

def dump_req_result_list(data: list[ReqResult], output_path: str, print_metrics: bool = True):
    """
    Dump the request result to a file
    
    The first line in the file contains the string representation of the list of ReqResult.

    If print_metrics is True, then metrics will be appended to the file for better human readability.
    """
    with open(output_path, "w") as f:
        f.write(str([dataclasses.asdict(d) for d in data]) + "\n")
        if print_metrics:
            from .metrics import BenchmarkMetrics
            metrics = BenchmarkMetrics.from_req_results(data)
            f.write(str(metrics))

def load_req_result_list(path: str) -> list[ReqResult]:
    with open(path, "r") as f:
        text = f.readline().strip()
        return [ReqResult(**d) for d in eval(text)]
