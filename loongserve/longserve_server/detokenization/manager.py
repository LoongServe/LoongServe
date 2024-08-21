import uvloop
import asyncio
import socket
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from ..io_struct import BatchTokenIdOut, ReqDetokenizationState, BatchStrOut, AbortReq, FinishStatus
from typing import Union
from .decode import decode_token
from ..tokenizer import get_tokenizer
import traceback

from loongserve.utils.infer_utils import calculate_time, mark_start, mark_end
from loongserve.utils.log_utils import init_logger

logger = init_logger(__name__)

class DeTokenizationManager:
    def __init__(
        self,
        detokenizer_id: int,
        model_weightdir: str,
        tokenizor_mode: str,
        httpserver_port: int,
        trust_remote_code: bool,
        skip_special_tokens: bool,
        spaces_between_special_tokens: bool,
    ):
        self.detokenizer_id = detokenizer_id

        context = zmq.asyncio.Context(2)

        self.send_to_httpserver = context.socket(zmq.PUSH)
        self.send_to_httpserver.connect(f"tcp://127.0.0.1:{httpserver_port}")   # HTTPServer, Router, and DeTokenizationManager are always co-located on one node

        self.tokenizer = get_tokenizer(model_weightdir, tokenizor_mode, trust_remote_code=trust_remote_code)
        self.req_id_to_out = {}

        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens

        print(f"Detokenization manager {self.detokenizer_id} init ok (on node {socket.gethostname()})")

    def handle_loop(self, recv_obj: Union[BatchTokenIdOut, ReqDetokenizationState, AbortReq]):
        assert isinstance(recv_obj, (BatchTokenIdOut, ReqDetokenizationState, AbortReq)), f"type is not right {type(recv_obj)}"
        if isinstance(recv_obj, ReqDetokenizationState):
            self.req_id_to_out[recv_obj.request_id] = recv_obj

        if isinstance(recv_obj, AbortReq):
            delete_req_id = recv_obj.req_id
            if delete_req_id in self.req_id_to_out:
                del self.req_id_to_out[delete_req_id]

        if isinstance(recv_obj, BatchTokenIdOut):
            new_batch_str_out = BatchStrOut()
            for req_id, new_token_id, new_gen_metadata, finish_status in recv_obj.reqs_infs:
                if req_id not in self.req_id_to_out:
                    continue
                req_out:ReqDetokenizationState = self.req_id_to_out[req_id]
                req_out.output_ids.append(new_token_id)
                req_out.gen_metadata.update(new_gen_metadata)

                out_text = decode_token(
                    self.tokenizer,
                    req_out,
                    new_token_id,
                    skip_special_tokens=self.skip_special_tokens,
                    spaces_between_special_tokens=self.spaces_between_special_tokens,
                )

                if out_text.endswith(u'\ufffd'):
                    new_text = ''
                else:
                    new_text = out_text[len(req_out.output_str):]
                    req_out.output_str = out_text
                new_batch_str_out.reqs_infs.append((req_id, new_text, new_gen_metadata, finish_status))
                if FinishStatus(finish_status).is_finished():
                    try:
                        del self.req_id_to_out[req_id]
                    except:
                        pass
            try:
                self.send_to_httpserver.send_pyobj(new_batch_str_out)
            except Exception as e:
                logger.error(f"detoken process has exception {str(e)}")
                traceback.print_exc()
                pass
