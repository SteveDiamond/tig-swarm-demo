"""
DFlash draft model architecture and training utilities.
DO NOT MODIFY — agents modify train.py, not this file.

Architecture based on the released DFlash inference code (github.com/z-lab/dflash).
Training utilities added for the reverse-engineering swarm.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# ---------------------------------------------------------------------------
# Architecture components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 1000000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)  # (bsz, 1, seq_len, dim)
    sin = sin.unsqueeze(1)
    q_len = q.size(2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashAttention(nn.Module):
    """
    Attention with KV injection from target model features.

    Q: from draft hidden states (q_len positions)
    K, V: concat of [target_context, draft_hidden] (ctx_len + q_len positions)

    The draft model attends to both injected target features and its own states.
    """

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden).view(bsz, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_ctx = self.v_proj(target_hidden).view(bsz, ctx_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_draft = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v_draft = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = torch.cat([self.k_norm(k_ctx), self.k_norm(k_draft)], dim=2)
        v = torch.cat([v_ctx, v_draft], dim=2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DFlashMLP(nn.Module):
    """SwiGLU MLP matching Qwen3 architecture."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_kv_heads: int, head_dim: int,
                 eps: float = 1e-6):
        super().__init__()
        self.self_attn = DFlashAttention(hidden_size, num_heads, num_kv_heads, head_dim, eps)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden, cos, sin, attention_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


@dataclass
class DFlashConfig:
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    head_dim: int = 128
    num_draft_layers: int = 5
    num_target_layers: int = 36
    num_target_features: int = 5
    block_size: int = 16
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 8192

    @classmethod
    def from_target(cls, target_config, num_draft_layers: int = 5,
                    num_target_features: int = 5, block_size: int = 16):
        return cls(
            hidden_size=target_config.hidden_size,
            intermediate_size=target_config.intermediate_size,
            num_attention_heads=target_config.num_attention_heads,
            num_key_value_heads=target_config.num_key_value_heads,
            head_dim=getattr(target_config, "head_dim",
                             target_config.hidden_size // target_config.num_attention_heads),
            num_draft_layers=num_draft_layers,
            num_target_layers=target_config.num_hidden_layers,
            num_target_features=num_target_features,
            block_size=block_size,
            vocab_size=target_config.vocab_size,
            rms_norm_eps=target_config.rms_norm_eps,
            rope_theta=getattr(target_config, "rope_theta", 1000000.0),
            max_position_embeddings=getattr(target_config, "max_position_embeddings", 8192),
        )


class DFlashDraftModel(nn.Module):
    """
    DFlash draft model for block diffusion speculative decoding.

    Takes noise embeddings (token embeddings for the block) and target context
    features (fused hidden states from the target model). Returns hidden states
    that are passed through the target model's LM head for logits.
    """

    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.target_layer_ids = build_target_layer_ids(
            config.num_target_layers, config.num_target_features,
        )
        self.layers = nn.ModuleList([
            DFlashDecoderLayer(
                config.hidden_size, config.intermediate_size,
                config.num_attention_heads, config.num_key_value_heads,
                config.head_dim, config.rms_norm_eps,
            )
            for _ in range(config.num_draft_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta,
        )
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size, bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noise_embedding: (bsz, block_size, hidden_size) — token embeddings for the block
            target_hidden: (bsz, ctx_len, num_features * hidden_size) — raw concatenated target features
            position_ids: (bsz, ctx_len + block_size) — positions for RoPE
                First ctx_len entries: context positions
                Last block_size entries: draft positions
            attention_mask: optional (bsz, 1, block_size, ctx_len + block_size) float mask

        Returns:
            (bsz, block_size, hidden_size) — hidden states for LM head
        """
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        cos, sin = self.rotary_emb(position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, target_hidden, cos, sin, attention_mask,
            )
        return self.norm(hidden_states)


# ---------------------------------------------------------------------------
# Target model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_features: int) -> List[int]:
    """Select layer IDs uniformly from layer 1 to num_target_layers - 3."""
    if num_features == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_features - 1)))
        for i in range(num_features)
    ]


def extract_context_features(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_ids: List[int],
) -> torch.Tensor:
    """
    Extract and concatenate hidden states from specified target model layers.

    Args:
        hidden_states: tuple of (bsz, seq_len, hidden_size) from target model
            with output_hidden_states=True. Index 0 is the embedding output.
        layer_ids: which transformer layers to extract (0-indexed into layers,
            offset by 1 to skip the embedding output)

    Returns:
        (bsz, seq_len, num_layers * hidden_size)
    """
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return torch.cat(selected, dim=-1)


def load_target_model(model_name: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load the frozen target model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def create_training_block(
    response_ids: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Sample a random anchor position and create a training block.

    Args:
        response_ids: (seq_len,) — token IDs for the response
        block_size: number of positions per block
        mask_token_id: token ID used for masked positions

    Returns:
        block_ids: (block_size,) — [anchor_token, mask, mask, ..., mask]
        labels: (block_size - 1,) — ground truth tokens at positions 1..block_size-1
        anchor_pos: int — position of the anchor in the response
    """
    max_anchor = len(response_ids) - block_size
    if max_anchor < 0:
        raise ValueError(f"Response too short ({len(response_ids)}) for block_size={block_size}")
    anchor_pos = torch.randint(0, max(1, max_anchor + 1), (1,)).item()

    block_ids = torch.full((block_size,), mask_token_id, dtype=response_ids.dtype)
    block_ids[0] = response_ids[anchor_pos]
    labels = response_ids[anchor_pos + 1 : anchor_pos + block_size]

    return block_ids, labels, anchor_pos


def build_block_position_ids(
    anchor_positions: List[int],
    prompt_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build position IDs for a batch of blocks.

    For each block starting at anchor position a (relative to response start):
    - Context positions: [prompt_len + a, ..., prompt_len + a + block_size - 1]
    - Draft positions: same (they represent the same sequence positions)
    - Total: 2 * block_size positions per block

    Args:
        anchor_positions: list of anchor positions (relative to response start)
        prompt_len: length of the prompt
        block_size: block size
        device: torch device

    Returns:
        (batch_size, 2 * block_size) position IDs
    """
    batch_size = len(anchor_positions)
    position_ids = torch.zeros(batch_size, 2 * block_size, dtype=torch.long, device=device)
    for i, a in enumerate(anchor_positions):
        base = prompt_len + a
        positions = torch.arange(base, base + block_size, device=device)
        position_ids[i, :block_size] = positions   # context positions
        position_ids[i, block_size:] = positions    # draft positions
    return position_ids


def compute_position_weights(block_size: int, gamma: float, device: torch.device) -> torch.Tensor:
    """
    Compute positional loss weights: w_k = exp(-(k-1)/gamma).

    Position k=1 (first prediction) gets weight 1.0.
    Later positions get exponentially decaying weights.

    Args:
        block_size: total block size (weights computed for positions 1..block_size-1)
        gamma: decay parameter (larger = more uniform)
        device: torch device

    Returns:
        (block_size - 1,) weights
    """
    k = torch.arange(1, block_size, dtype=torch.float32, device=device)
    return torch.exp(-(k - 1) / gamma)


# ---------------------------------------------------------------------------
# Speculative decoding for evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def speculative_decode(
    target_model: nn.Module,
    draft_model: DFlashDraftModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    mask_token_id: int,
    temperature: float = 0.0,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Run speculative decoding and return generated tokens + acceptance lengths.

    Args:
        target_model: frozen target LLM
        draft_model: trained draft model
        input_ids: (1, prompt_len) input token IDs
        max_new_tokens: maximum tokens to generate
        block_size: draft block size
        mask_token_id: token ID for mask positions
        temperature: sampling temperature (0 = greedy)

    Returns:
        output_ids: (1, total_len) generated sequence
        acceptance_lengths: list of accepted lengths per block
    """
    device = input_ids.device
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens

    output_ids = torch.full((1, max_length + block_size), mask_token_id,
                            dtype=torch.long, device=device)
    output_ids[:, :num_input] = input_ids

    target_out = target_model(
        input_ids, output_hidden_states=True, use_cache=False,
    )
    first_token = _sample(target_out.logits[:, -1:, :], temperature)
    output_ids[:, num_input] = first_token.squeeze()

    target_features = extract_context_features(
        target_out.hidden_states, draft_model.target_layer_ids,
    )

    acceptance_lengths = []
    start = num_input

    embed_fn = target_model.model.embed_tokens
    lm_head = target_model.lm_head
    layer_ids = draft_model.target_layer_ids

    while start < max_length:
        end = min(start + block_size, max_length)
        actual_bs = end - start
        block_tokens = output_ids[:, start:end].clone()

        if actual_bs > 1:
            noise_emb = embed_fn(block_tokens)
            ctx_features = target_features[:, start:end, :]

            pos_ids = torch.arange(start, start + 2 * actual_bs, device=device).unsqueeze(0)
            ctx_pos = torch.arange(start, end, device=device)
            draft_pos = torch.arange(start, end, device=device)
            pos_ids = torch.cat([ctx_pos, draft_pos]).unsqueeze(0)

            draft_hidden = draft_model(
                noise_embedding=noise_emb,
                target_hidden=ctx_features,
                position_ids=pos_ids,
            )
            draft_logits = lm_head(draft_hidden[:, 1:, :])
            block_tokens[:, 1:actual_bs] = _sample(draft_logits[:, :actual_bs-1, :], temperature)

        target_out = target_model(
            block_tokens, output_hidden_states=True, use_cache=False,
        )
        posterior = _sample(target_out.logits, temperature)

        accepted = (block_tokens[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start:start + accepted + 1] = block_tokens[:, :accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[:, accepted]
        start += accepted + 1
        acceptance_lengths.append(accepted + 1)

        if actual_bs > 1:
            target_features = extract_context_features(
                target_out.hidden_states, layer_ids,
            )[:, :accepted + 1, :]
            new_features = torch.full(
                (1, max_length + block_size - target_features.shape[1],
                 target_features.shape[2]),
                0, dtype=target_features.dtype, device=device,
            )
            target_features = torch.cat([
                extract_context_features(
                    target_model(
                        output_ids[:, :start],
                        output_hidden_states=True, use_cache=False,
                    ).hidden_states,
                    layer_ids,
                ),
            ], dim=1)

    output_ids = output_ids[:, :min(start + 1, max_length)]
    return output_ids, acceptance_lengths


def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])


# ---------------------------------------------------------------------------
# Simplified evaluation (no KV cache, re-runs target each block)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_acceptance_length(
    target_model: nn.Module,
    draft_model: DFlashDraftModel,
    prompts: List[torch.Tensor],
    block_size: int,
    mask_token_id: int,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
) -> dict:
    """
    Evaluate draft model quality by measuring acceptance length.

    Returns dict with:
        mean_accepted_length: average tokens accepted per block
        per_position_accuracy: accuracy at each position in the block
        num_prompts: number of prompts evaluated
    """
    all_acceptance = []
    per_position_correct = torch.zeros(block_size, device=next(draft_model.parameters()).device)
    per_position_total = torch.zeros(block_size, device=next(draft_model.parameters()).device)

    embed_fn = target_model.model.embed_tokens
    lm_head = target_model.lm_head
    layer_ids = draft_model.target_layer_ids

    for prompt_ids in prompts:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        device = prompt_ids.device

        target_out = target_model(
            prompt_ids, output_hidden_states=True, use_cache=False,
        )
        greedy_first = torch.argmax(target_out.logits[:, -1:, :], dim=-1)
        target_features = extract_context_features(
            target_out.hidden_states, layer_ids,
        )

        generated = torch.cat([prompt_ids, greedy_first], dim=1)
        n_blocks = max_new_tokens // block_size

        for _ in range(n_blocks):
            pos = generated.shape[1]
            block_tokens = torch.full((1, block_size), mask_token_id,
                                      dtype=torch.long, device=device)
            block_tokens[:, 0] = generated[:, -1]

            full_out = target_model(generated, output_hidden_states=True, use_cache=False)
            target_features = extract_context_features(full_out.hidden_states, layer_ids)

            noise_emb = embed_fn(block_tokens)
            ctx = target_features[:, -block_size:, :] if target_features.shape[1] >= block_size \
                else F.pad(target_features, (0, 0, block_size - target_features.shape[1], 0))

            ctx_positions = torch.arange(
                max(0, pos - block_size), pos, device=device,
            )
            if len(ctx_positions) < block_size:
                ctx_positions = F.pad(ctx_positions, (block_size - len(ctx_positions), 0))
            draft_positions = torch.arange(pos - 1, pos - 1 + block_size, device=device)
            position_ids = torch.cat([ctx_positions, draft_positions]).unsqueeze(0)

            draft_hidden = draft_model(
                noise_embedding=noise_emb,
                target_hidden=ctx[:, :block_size, :],
                position_ids=position_ids,
            )
            draft_logits = lm_head(draft_hidden[:, 1:, :])
            draft_tokens = _sample(draft_logits, temperature)
            block_tokens[:, 1:] = draft_tokens

            verify_input = block_tokens
            verify_out = target_model(
                torch.cat([generated[:, :-1], block_tokens], dim=1),
                output_hidden_states=True, use_cache=False,
            )
            verify_logits = verify_out.logits[:, -(block_size):, :]
            verify_tokens = torch.argmax(verify_logits, dim=-1)

            for k in range(block_size):
                per_position_total[k] += 1
                if block_tokens[0, k] == verify_tokens[0, k]:
                    per_position_correct[k] += 1

            accepted = (block_tokens[0, 1:] == verify_tokens[0, :-1]).int()
            acc_len = 0
            for a in accepted:
                if a == 1:
                    acc_len += 1
                else:
                    break
            all_acceptance.append(acc_len + 1)

            next_token = verify_tokens[:, min(acc_len, block_size - 1):][:, :1]
            generated = torch.cat([generated, block_tokens[:, :acc_len + 1], next_token], dim=1)

            if generated.shape[1] > prompt_ids.shape[1] + max_new_tokens:
                break

    per_pos_acc = (per_position_correct / per_position_total.clamp(min=1)).tolist()

    return {
        "mean_accepted_length": sum(all_acceptance) / max(len(all_acceptance), 1),
        "per_position_accuracy": per_pos_acc,
        "total_blocks": len(all_acceptance),
        "num_prompts": len(prompts),
    }
