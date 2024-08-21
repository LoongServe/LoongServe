class LongServeLlamaPreAndPostLayerWeight:
    def __init__(self, total_rank, total_world_size, tp_rank, tp_world_size, sp_rank, sp_world_size, data_type, network_config, mode):
        self.total_rank_ = total_rank
        self.total_world_size_ = total_world_size
        self.tp_rank_ = tp_rank
        self.tp_world_size_ = tp_world_size
        self.sp_rank_ = sp_rank
        self.sp_world_size_ = sp_world_size

        self.data_type_ = data_type
        self.network_config_ = network_config
        self.mode = mode
        self.init_static_params()
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_vob_size = vob_size // self.tp_world_size_
        n_embed = self.network_config_["hidden_size"]
        if "model.embed_tokens.weight" in weights:
            self.wte_weight_ = self._cuda(weights['model.embed_tokens.weight'][split_vob_size *
                                                                    self.tp_rank_: split_vob_size * (self.tp_rank_ + 1), :])
        if 'lm_head.weight' in weights:
            self.lm_head_weight_ = self._cuda(weights['lm_head.weight'][split_vob_size * self.tp_rank_: split_vob_size *
                                                            (self.tp_rank_ + 1), :])
        if 'model.norm.weight' in weights:
            self.final_norm_weight_ = self._cuda(weights['model.norm.weight'])

        return

    def init_static_params(self):
        """
        design for some static init params, many model dont need do this.
        """
        pass

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_, 
                   self.lm_head_weight_, 
                   self.final_norm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return 

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda()