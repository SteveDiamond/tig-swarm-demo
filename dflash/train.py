"""
DFlash draft model training script.
THIS IS THE FILE YOU MODIFY — experiment with hyperparameters, training
strategies, loss functions, and optimization to maximize acceptance length.

Usage: uv run train.py
"""

import os
import gc
import json
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from model import (
    DFlashConfig,
    DFlashDraftModel,
    extract_context_features,
    load_target_model,
    create_training_block,
    build_block_position_ids,
    compute_position_weights,
)
from prepare import CACHE_DIR, TARGET_MODEL, TRAIN_DATA_PATH, EVAL_DATA_PATH

# ---------------------------------------------------------------------------
# Hyperparameters (edit these freely)
# ---------------------------------------------------------------------------

# Architecture
NUM_DRAFT_LAYERS = 5          # number of draft transformer layers
NUM_TARGET_FEATURES = 5       # number of target model layers to extract features from
BLOCK_SIZE = 16               # tokens per block (paper uses 16 for Qwen, 10 for LLaMA)

# Optimization
LR = 3e-4                    # peak learning rate
OPTIMIZER = "adamw"           # optimizer: "adamw", "adam", "sgd"
BETAS = (0.9, 0.999)         # Adam betas
WEIGHT_DECAY = 0.0            # weight decay
WARMUP_STEPS = 100            # LR warmup steps
LR_SCHEDULE = "constant"     # "constant", "cosine", "linear"
GRAD_CLIP = 1.0               # max gradient norm (0 to disable)
BATCH_SIZE = 4                # sequences per step (limited by VRAM)

# Loss
GAMMA = 4.0                  # positional weight decay (w_k = exp(-(k-1)/gamma))
LABEL_SMOOTHING = 0.0         # label smoothing for cross-entropy

# Training
NUM_STEPS = 2000              # total training steps
BLOCKS_PER_SEQ = 1            # training blocks sampled per sequence
SEED = 42                     # random seed

# EMA
USE_EMA = False               # use exponential moving average
EMA_DECAY = 0.999             # EMA decay rate

# Mask token
MASK_TOKEN_ID = 0             # token ID used for masked positions

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.cuda.manual_seed(SEED)

use_amp = device.type == "cuda"
amp_dtype = torch.bfloat16

# ---------------------------------------------------------------------------
# Load target model (frozen)
# ---------------------------------------------------------------------------

print(f"Loading target model: {TARGET_MODEL}")
target_model, tokenizer = load_target_model(
    TARGET_MODEL, device=device, dtype=amp_dtype,
)
target_config = target_model.config
print(f"Target: {target_config.num_hidden_layers} layers, "
      f"hidden_size={target_config.hidden_size}, "
      f"vocab_size={target_config.vocab_size}")

# ---------------------------------------------------------------------------
# Create draft model
# ---------------------------------------------------------------------------

draft_config = DFlashConfig.from_target(
    target_config,
    num_draft_layers=NUM_DRAFT_LAYERS,
    num_target_features=NUM_TARGET_FEATURES,
    block_size=BLOCK_SIZE,
)
draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=amp_dtype)

num_params = sum(p.numel() for p in draft_model.parameters() if p.requires_grad)
print(f"Draft model: {NUM_DRAFT_LAYERS} layers, {num_params / 1e6:.1f}M trainable parameters")
print(f"Block size: {BLOCK_SIZE}, Target features from layers: {draft_model.target_layer_ids}")

# EMA
ema_model = None
if USE_EMA:
    from copy import deepcopy
    ema_model = deepcopy(draft_model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

# ---------------------------------------------------------------------------
# Load training data
# ---------------------------------------------------------------------------

print(f"Loading training data from {TRAIN_DATA_PATH}")
train_data = torch.load(TRAIN_DATA_PATH, weights_only=True)
all_input_ids = train_data["input_ids"]
all_prompt_lens = train_data["prompt_lens"]
print(f"Training sequences: {len(all_input_ids)}")

# Shared embedding and LM head from target (frozen)
embed_fn = target_model.model.embed_tokens
lm_head = target_model.lm_head

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

if OPTIMIZER == "adamw":
    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=LR, betas=BETAS,
                                  weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(draft_model.parameters(), lr=LR, betas=BETAS)
elif OPTIMIZER == "sgd":
    optimizer = torch.optim.SGD(draft_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

position_weights = compute_position_weights(BLOCK_SIZE, GAMMA, device)

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    if LR_SCHEDULE == "constant":
        return LR
    elif LR_SCHEDULE == "cosine":
        progress = (step - WARMUP_STEPS) / max(NUM_STEPS - WARMUP_STEPS, 1)
        return LR * 0.5 * (1 + math.cos(math.pi * progress))
    elif LR_SCHEDULE == "linear":
        progress = (step - WARMUP_STEPS) / max(NUM_STEPS - WARMUP_STEPS, 1)
        return LR * (1 - progress)
    return LR


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print(f"\nStarting training: {NUM_STEPS} steps, batch_size={BATCH_SIZE}")
print(f"Optimizer: {OPTIMIZER}, LR: {LR}, Schedule: {LR_SCHEDULE}")
print(f"Gamma (loss decay): {GAMMA}, Label smoothing: {LABEL_SMOOTHING}")
print("-" * 60)

draft_model.train()
t_start = time.time()
smooth_loss = 0.0
best_loss = float("inf")
losses = []
data_indices = list(range(len(all_input_ids)))

for step in range(NUM_STEPS):
    t0 = time.time()

    # --- Sample batch ---
    batch_indices = random.choices(data_indices, k=BATCH_SIZE)

    total_loss = 0.0
    num_tokens = 0

    for idx in batch_indices:
        input_ids = all_input_ids[idx].to(device)
        prompt_len = all_prompt_lens[idx]
        response_ids = input_ids[prompt_len:]

        if len(response_ids) < BLOCK_SIZE + 1:
            continue

        # --- Forward through target model (frozen, no grad) ---
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            target_out = target_model(
                input_ids.unsqueeze(0),
                output_hidden_states=True,
                use_cache=False,
            )
            target_features = extract_context_features(
                target_out.hidden_states,
                draft_model.target_layer_ids,
            )

        # --- Create training block ---
        block_ids, labels, anchor_pos = create_training_block(
            response_ids, BLOCK_SIZE, MASK_TOKEN_ID,
        )

        abs_anchor = prompt_len + anchor_pos
        block_ids = block_ids.to(device)
        labels = labels.to(device)

        # --- Get embeddings and context ---
        with torch.no_grad():
            noise_emb = embed_fn(block_ids.unsqueeze(0))
            ctx_features = target_features[:, abs_anchor:abs_anchor + BLOCK_SIZE, :]

        # --- Position IDs ---
        ctx_positions = torch.arange(abs_anchor, abs_anchor + BLOCK_SIZE, device=device)
        draft_positions = ctx_positions.clone()
        position_ids = torch.cat([ctx_positions, draft_positions]).unsqueeze(0)

        # --- Forward draft model ---
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            draft_hidden = draft_model(
                noise_embedding=noise_emb,
                target_hidden=ctx_features,
                position_ids=position_ids,
            )
            logits = lm_head(draft_hidden[:, 1:, :])  # skip anchor, predict B-1 tokens

            # --- Weighted cross-entropy loss ---
            loss_per_pos = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
                label_smoothing=LABEL_SMOOTHING,
            )
            weights = position_weights[:len(labels)]
            weighted_loss = (loss_per_pos * weights).sum() / weights.sum()

        total_loss += weighted_loss
        num_tokens += len(labels)

    if num_tokens == 0:
        continue

    batch_loss = total_loss / BATCH_SIZE

    # --- Backward + optimize ---
    optimizer.zero_grad(set_to_none=True)
    batch_loss.backward()

    if GRAD_CLIP > 0:
        torch.nn.utils.clip_grad_norm_(draft_model.parameters(), GRAD_CLIP)

    lr = get_lr(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()

    # --- EMA update ---
    if ema_model is not None:
        with torch.no_grad():
            for sp, mp in zip(ema_model.parameters(), draft_model.parameters()):
                sp.lerp_(mp, 1 - EMA_DECAY)

    # --- Logging ---
    lv = batch_loss.item()
    smooth_loss = 0.9 * smooth_loss + 0.1 * lv if step > 0 else lv
    losses.append(lv)
    dt = time.time() - t0

    if step % 10 == 0:
        elapsed = time.time() - t_start
        print(f"step {step:05d} | loss: {smooth_loss:.4f} | lr: {lr:.2e} | "
              f"dt: {dt*1000:.0f}ms | elapsed: {elapsed:.0f}s")

    # --- Save best checkpoint ---
    if smooth_loss < best_loss and step > 20:
        best_loss = smooth_loss
        torch.save({
            "step": step,
            "model_state": draft_model.state_dict(),
            "config": draft_config,
            "loss": best_loss,
        }, CACHE_DIR / "draft_checkpoint_best.pt")

    if step == 0:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Save final checkpoint
# ---------------------------------------------------------------------------

t_end = time.time()
training_seconds = t_end - t_start

save_model = ema_model if ema_model is not None else draft_model
torch.save({
    "step": step,
    "model_state": save_model.state_dict(),
    "config": draft_config,
    "loss": smooth_loss,
}, CACHE_DIR / "draft_checkpoint_final.pt")

# ---------------------------------------------------------------------------
# Training summary
# ---------------------------------------------------------------------------

peak_vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0

print()
print("=" * 60)
print("Training complete")
print("=" * 60)
print(f"training_seconds: {training_seconds:.1f}")
print(f"final_loss:       {smooth_loss:.6f}")
print(f"best_loss:        {best_loss:.6f}")
print(f"total_steps:      {step + 1}")
print(f"peak_vram_gb:     {peak_vram:.1f}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"draft_layers:     {NUM_DRAFT_LAYERS}")
print(f"block_size:       {BLOCK_SIZE}")
print(f"gamma:            {GAMMA}")
print(f"lr:               {LR}")
print(f"optimizer:        {OPTIMIZER}")
print(f"lr_schedule:      {LR_SCHEDULE}")
print(f"checkpoint:       {CACHE_DIR / 'draft_checkpoint_final.pt'}")
