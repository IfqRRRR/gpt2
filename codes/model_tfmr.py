import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

ACT2FN = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "linear": lambda x: x,
    "sigmoid": torch.sigmoid,
    "gelu": F.gelu,
}

class TransposeLinear(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class TfmrAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            # TODO START
            # define the bias term for constructing the causal mask (i.e., seeing only prefix tokens).
            torch.tril(torch.ones([max_positions, max_positions])).view(1, 1, max_positions, max_positions)
            # TODO END
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.c_attn = TransposeLinear(3 * self.embed_dim, self.embed_dim)
        self.c_proj = TransposeLinear(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value):     # [32, 12, 36, 64]      use_cache: [32, 12, 1(fixed), 64], [32, 12, 2, 64], [32, 12, 2, 64]
        # TODO START
        # implement the multi-head mask self-attnetion mechanism
        attn_weights = torch.matmul(query, key.transpose(-1, -2))       # [32, 12, 36, 36]      use_cache: [32, 12, 1, 2]
        if self.scale_attn_weights:
            attn_weights /= (float(value.size(-1)) ** 0.5)
        
        query_length, key_length = query.size(-2), key.size(-2)
        mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()   # [1, 1, 36, 36]的下三角矩阵
        attn_weights = torch.where(mask, attn_weights, self.masked_bias)  # [32, 12, 36, 36]
        attn_weights = F.softmax(attn_weights, dim=-1)  # [32, 12, 36, 36]
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)  # [32, 12, 36, 64]

        return attn_output, attn_weights
        # TODO END

    def _split_heads(self, tensor, num_heads, attn_head_size):
        # TODO START
        # Splits hidden_size dim into attn_head_size and num_heads
        tensor = tensor.view(tensor.shape[0], tensor.shape[1], num_heads, attn_head_size)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        # TODO END

    def _merge_heads(self, tensor, num_heads, attn_head_size):  # [32, 12, 36, 64]
        # TODO START
        # Merges attn_head_size dim and num_attn_heads dim into hidden_size
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(tensor.shape[0], tensor.shape[1], num_heads * attn_head_size)
        # TODO END

    def forward(
        self,
        hidden_states,      # [32, 36, 768]
        layer_past=None,
        use_cache=False,
    ):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)    # [32, 36, 768] [32, 36, 768] [32, 36, 768]
        
        query = self._split_heads(query, self.num_heads, self.head_dim)     # [32, 12, 36, 64]
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:  # [32, 12, n += 1, 64]
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value)   # [32, 12, 36, 64]  [32, 12, 36, 36]    weights为key和query的乘积

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)     # [32, 36, 768]
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)     # [32, 36, 768]

        outputs = (attn_output, present)
        outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class TfmrMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = TransposeLinear(intermediate_size, embed_dim)
        self.c_proj = TransposeLinear(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TfmrBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = TfmrAttention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = TfmrMLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        use_cache=False,
    ):
        residual = hidden_states        # [32, 36, 768]
        hidden_states = self.ln_1(hidden_states)    # [32, 36, 768]
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]      # use_cache: key, value + attn_wieght (query * key)

        # TODO START
        # Bulid connecetions of different modules in the Tranformer block
        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        # TODO END

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class TfmrModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([TfmrBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
    
    def discard_layer(self, args):
        self.choose_layer = args.choose_layer.split(' ')
        self.temp_h =  nn.ModuleList([])
        for i, h in enumerate(self.h):
            if str(i + 1) in self.choose_layer:
                self.temp_h.append(h)
        self.h = self.temp_h
        
    def get_input_embeddings(self):
        return self.wte

    def forward(
        self,
        input_ids,
        past_key_values=None,
        use_cache=None,
    ):
        input_shape = input_ids.size()  # [32, 36]
        input_ids = input_ids.view(-1, input_shape[-1])

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        inputs_embeds = self.wte(input_ids) # [32, 36, 768]
        # TODO START
        # Implement the positional embeddings. Note that the length of cache hidden states used during inference
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].shape[-2]   # 从1开始每次加一

        position_ids = torch.arange(past_length, input_shape[-1] + past_length).to(device)
        position_ids = position_ids.view(-1, input_shape[-1])
        position_embeds = self.wpe(position_ids) # [32, 36, 768]
        # TODO END
        hidden_states = inputs_embeds + position_embeds # [32, 36, 768]

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)  # [32, 36, 768]

        presents = () if use_cache else None
        all_self_attentions = ()    # unuse
        all_cross_attentions = ()    # unuse
        all_hidden_states = ()    # unuse
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                use_cache=use_cache,
            )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)     # outputs[1]: (key, value)

            all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        all_hidden_states = all_hidden_states + (hidden_states,)

        return {
            "last_hidden_state": hidden_states,     # [32, 36, 768]
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions,
        }


class TfmrLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TfmrModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids,
        past_key_values=None,
        labels=None,
        use_cache=None,
        PAD_ID=None,
    ):
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs["last_hidden_state"]    # [32, 36, 768]
        lm_logits = self.lm_head(hidden_states) # [32, 36, 50257]

        loss = None
        if labels is not None:
            ce_loss_fct = CrossEntropyLoss(reduction="none")
            # TODO START
            # Implement the loss function. Note that you should shift logits so that tokens < n predict n
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_mask = (shift_labels[:, :-1] != PAD_ID)    # last one is discarded
            first_mask = torch.ones([shift_labels.shape[0], 1]).to(shift_labels.device)
            loss_mask = torch.cat([first_mask, loss_mask], dim=1)   # [32, 35]
            loss = ce_loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))
            loss = loss.view(shift_logits.shape[0], -1)     # [32, 35]
            loss = ((loss * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1) + 1e-10)).mean()
            # TODO END

        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": transformer_outputs["past_key_values"],
            "hidden_states": transformer_outputs["hidden_states"],
            "attentions": transformer_outputs["attentions"],
            "cross_attentions": transformer_outputs["cross_attentions"],
         }
        

    def inference(self, device, PAD_ID, batch_size, maxlen, decode_strategy, temperature, top_p=1.0, top_k=50267):
        self.eval()
        allgen = []
        with torch.no_grad():
            for i in range(0, int(5000/batch_size)+1):
                input_ids = torch.tensor([[PAD_ID] for _ in range(batch_size)]).to(device)
                past_key_values = None
                output_ids = input_ids
                for _ in range(maxlen):
                    outputs = self(input_ids, past_key_values=past_key_values, use_cache=True)
                    logits = outputs["logits"]
                    past_key_values = outputs["past_key_values"]
                    logits = logits[:, -1, :] / temperature
                    if decode_strategy == "top-p":
                        # TODO START
                        # implement top-p sampling
                        sorted_logits, sorted_indexes = torch.sort(logits, descending=True)
                        prob_logits = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(prob_logits, dim=-1)

                        temp_indexes_to_remove = cumulative_probs > top_p
                        zero_mask = torch.zeros([temp_indexes_to_remove.shape[0], 1]).to(temp_indexes_to_remove.device)
                        indexes_to_remove = torch.cat([zero_mask, temp_indexes_to_remove[:, :-1]], dim=1).bool()  # 第一个补零，后面去掉一个
                        
                        sorted_indexes = (sorted_indexes + torch.arange(sorted_indexes.shape[0], device=device, dtype=torch.long).unsqueeze(-1) * sorted_indexes.shape[1])

                        indices_to_remove = torch.masked_select(sorted_indexes, indexes_to_remove)
                        logits = logits.reshape(-1)
                        logits[indices_to_remove] = -float("inf")
                        logits = logits.reshape(sorted_indexes.shape[0], sorted_indexes.shape[1])
                        # TODO END
                    elif decode_strategy == "top-k":
                        # TODO START
                        # implement top-k sampling
                        sorted_logits, sorted_indexes = torch.sort(logits, descending=True)
                        single_indexes_remove = torch.ones(sorted_indexes.shape[1]).to(device)
                        single_indexes_remove[:top_k] = torch.zeros(top_k)
                        indexes_to_remove = single_indexes_remove.repeat(sorted_indexes.shape[0], 1).bool()
                        sorted_indexes = (sorted_indexes + torch.arange(sorted_indexes.shape[0], device=device, dtype=torch.long).unsqueeze(-1) * sorted_indexes.shape[1])

                        indices_to_remove = torch.masked_select(sorted_indexes, indexes_to_remove)
                        logits = logits.reshape(-1)
                        logits[indices_to_remove] = -float("inf")
                        logits = logits.reshape(sorted_indexes.shape[0], sorted_indexes.shape[1])
                        # TODO END
                    prob = logits.softmax(dim=-1) # shape: (batch_size, num_vocabs)
                    now_token = torch.multinomial(prob, 1)[:, :1] # shape: (batch_size)

                    output_ids = torch.cat([output_ids, now_token], 1)
                    input_ids = now_token
                allgen += output_ids.cpu().numpy().tolist()
        pro_allgen = []
        for gen in allgen[:5000]:
            pro_allgen.append([])
            for idx in gen[1:]:
                if idx == PAD_ID:
                    break
                pro_allgen[-1].append(idx)
        self.train() # return to training mode
        return pro_allgen
                