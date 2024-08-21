import socket
from loongserve.longserve_server.tokenizer import get_tokenizer

class TokenizationManager:
    def __init__(
        self,
        tokenizer_id,
        args
    ) -> None:
        self.tokenizer_id = tokenizer_id
        
        self.args = args
        self.tokenizer = get_tokenizer(
            args.model_dir, args.tokenizer_mode, trust_remote_code=args.trust_remote_code
        )

        print(f"Tokenization manager {self.tokenizer_id} init ok (on node {socket.gethostname()})", flush=True)
    
    def encode(self, prompt):
        return self.tokenizer.encode(prompt)