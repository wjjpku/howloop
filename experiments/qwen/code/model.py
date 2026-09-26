"""Shared whole-stack Qwen3 recurrence, with unchanged token-position RoPE.

No module-list mutation, hidden-state detach, recurrent KV cache, extra loop
embedding, or per-loop final normalization. The LM head is applied only once.
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class DenseAffine(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width, width, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(width, dtype=torch.float32))

    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            delta = torch.nn.functional.linear(hidden.float(), self.weight, self.bias)
        return hidden + delta.to(hidden.dtype)


class LoopedQwen3(nn.Module):
    def __init__(self, base, loops=4, train_j=False, activation_checkpointing=True):
        super().__init__()
        if base.config.model_type != "qwen3":
            raise ValueError("This experiment requires the dense Qwen3 architecture")
        if getattr(base.model, "has_sliding_layers", False):
            raise ValueError("Sliding attention is outside the fixed protocol")
        self.base = base
        self.loops = loops
        self.activation_checkpointing = activation_checkpointing
        self.controller = DenseAffine(base.config.hidden_size) if train_j else None
        self.use_j = train_j
        self.base.config.use_cache = False
        # Explicit checkpointing below avoids nested HF checkpoint wrappers.
        self.base.gradient_checkpointing_disable()
        for parameter in self.base.parameters():
            parameter.requires_grad_(not train_j)

    def distribute(self, devices):
        """Single-process layer model-parallelism; no weight replicas or DTensors."""
        self.devices = [torch.device(d) for d in devices]
        self.base.model.embed_tokens.to(self.devices[0])
        self.base.model.rotary_emb.to(self.devices[0])
        depth = len(self.base.model.layers)
        for index, layer in enumerate(self.base.model.layers):
            layer.to(self.devices[min(len(devices) - 1, index * len(devices) // depth)])
        self.base.model.norm.to(self.devices[-1])
        self.base.lm_head.to(self.devices[-1])
        if self.controller is not None:
            self.controller.to(self.devices[0])
        return self

    def forward(self, input_ids, attention_mask=None, *, loops=None,
                labels=None, return_boundaries=False, last_only=False):
        loops = self.loops if loops is None else int(loops)
        if loops < 1:
            raise ValueError("loops must be positive")
        hidden = self.base.model.embed_tokens(input_ids)
        if hidden.device.type == "cuda" and torch.is_autocast_enabled("cuda"):
            hidden = hidden.to(torch.get_autocast_dtype("cuda"))
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        length = input_ids.shape[1]
        cache_position = torch.arange(length, device=input_ids.device)
        # Boolean SDPA mask: True means allowed. Query padding never contributes
        # to the objective; future and padded keys are excluded at every loop.
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        causal = causal[None, None] & attention_mask[:, None, None, :].bool()
        position_embeddings = self.base.model.rotary_emb(hidden, position_ids)
        device_kwargs = {}
        boundaries = []
        for loop in range(loops):
            if loop and self.use_j:
                if self.controller is None:
                    raise RuntimeError("J requested but no controller exists")
                hidden = self.controller(hidden.to(self.controller.weight.device))
            for layer in self.base.model.layers:
                device = next(layer.parameters()).device
                hidden = hidden.to(device)
                if device not in device_kwargs:
                    device_kwargs[device] = dict(attention_mask=causal.to(device),
                        position_ids=position_ids.to(device), past_key_value=None,
                        use_cache=False, cache_position=cache_position.to(device),
                        position_embeddings=tuple(p.to(device) for p in position_embeddings))
                layer_kwargs = device_kwargs[device]
                if self.training and self.activation_checkpointing and torch.is_grad_enabled():
                    hidden = checkpoint(layer, hidden, use_reentrant=False, **layer_kwargs)
                else:
                    hidden = layer(hidden, **layer_kwargs)
            if return_boundaries:
                boundaries.append(hidden)
                if hidden.requires_grad:
                    hidden.retain_grad()
        hidden = self.base.model.norm(hidden.to(self.base.model.norm.weight.device))
        if last_only:
            hidden = hidden[:, -1:, :]
        logits = self.base.lm_head(hidden)
        loss = None
        if labels is not None:
            if last_only:
                raise ValueError("last_only is only for generation")
            target = labels[:, 1:].to(logits.device)
            mask = target.ne(-100)
            token_loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().transpose(1, 2), target,
                ignore_index=-100, reduction="none")
            # Equal weight per answer / sequence, not per digit.
            loss = ((token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
        return SimpleNamespace(logits=logits, loss=loss, boundaries=boundaries)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens, eos_token_id,
                 pad_token_id, loops=None):
        # Single-process model parallelism allows an early stop after all EOS.
        ids, mask = input_ids, attention_mask
        finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens):
            scores = self(ids, mask, loops=loops, last_only=True).logits[:, -1].to(ids.device)
            next_id = scores.argmax(-1)
            next_id = torch.where(finished, pad_token_id, next_id)
            ids = torch.cat((ids, next_id[:, None]), 1)
            mask = torch.cat((mask, (~finished).long()[:, None]), 1)
            finished |= next_id.eq(eos_token_id)
            if bool(finished.all()):
                break
        return ids
