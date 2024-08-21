import threading
import torch

class LongServeLlamaTransformerLayerWeight:
    def __init__(self, layer_num, total_rank, total_world_size, tp_rank, tp_world_size, sp_rank, sp_world_size, data_type, network_config, mode=[]):
        self.layer_num_ = layer_num
        self.total_rank_ = total_rank
        self.total_world_size_ = total_world_size
        self.tp_rank_ = tp_rank
        self.tp_world_size_ = tp_world_size
        self.sp_rank_ = sp_rank
        self.sp_world_size_ = sp_world_size
        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.lock = threading.Lock()
        self.init_static_params()
        return
    
    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.kv_weight_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.gate_up_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 
    
    def _try_cat_to(self, source_tensor_names, dest_name, cat_dim, handle_func=None):
        if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
            with self.lock:
                if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
                    tensors = [getattr(self, name, None) for name in source_tensor_names]
                    ans = torch.cat(tensors, dim=cat_dim)
                    if handle_func is not None:
                        ans = handle_func(ans)
                    else:
                        ans = self._cuda(ans)
                    setattr(self, dest_name, ans)
                    for name in source_tensor_names:
                        delattr(self, name)
        return
    
    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.tp_world_size_
        kv_split_n_embed = n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"] // self.tp_world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            self.q_weight_ = self.q_weight_[q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"]
            k_weight_ = k_weight_[kv_split_n_embed * self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = k_weight_.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"]
            v_weight_ = v_weight_[kv_split_n_embed * self.tp_rank_: kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = v_weight_.transpose(0, 1)
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_: q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        
        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)
        
        return
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.tp_world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = up_proj.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = gate_proj.transpose(0, 1)
        
        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1)

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
        return
    
    def _cuda(self, cpu_tensor):
        # if self.total_rank_ is None:
        return cpu_tensor.cuda().to(self.data_type_).contiguous()
        # else:
            # return cpu_tensor.cuda(self.total_rank_).to(self.data_type_).contiguous()