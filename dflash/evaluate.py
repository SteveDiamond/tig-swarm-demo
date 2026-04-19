"""
DFlash draft model evaluation harness.
Measures acceptance length via speculative decoding on eval prompts.

Usage: uv run evaluate.py [--checkpoint PATH] [--tier 1|2]

DO NOT MODIFY — agents modify train.py, not this file.
"""

import json
import sys
import time
import torch
import torch.nn.functional as F
from pathlib import Path

from model import (
    DFlashConfig,
    DFlashDraftModel,
    extract_context_features,
    load_target_model,
    evaluate_acceptance_length,
    _sample,
)
from prepare import CACHE_DIR, TARGET_MODEL, EVAL_DATA_PATH


def load_draft_model(checkpoint_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load a trained draft model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = DFlashDraftModel(config).to(device=device, dtype=dtype)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


@torch.inference_mode()
def tier1_eval(
    target_model, draft_model, eval_data, block_size, mask_token_id, device, dtype,
):
    """
    Tier 1: Fast proxy evaluation.
    Measures per-position prediction accuracy against target model's greedy output.
    No speculative decoding — just checks if draft predictions match target.
    """
    embed_fn = target_model.model.embed_tokens
    lm_head = target_model.lm_head
    layer_ids = draft_model.target_layer_ids

    per_pos_correct = torch.zeros(block_size - 1, device=device)
    per_pos_total = torch.zeros(block_size - 1, device=device)
    total_blocks = 0

    for prompt_ids in eval_data:
        prompt_ids = prompt_ids.to(device).unsqueeze(0)

        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            target_out = target_model(
                prompt_ids, output_hidden_states=True, use_cache=False,
            )

        target_features = extract_context_features(
            target_out.hidden_states, layer_ids,
        )

        greedy_first = torch.argmax(target_out.logits[:, -1:, :], dim=-1)
        seq_len = prompt_ids.shape[1]

        block_tokens = torch.full((1, block_size), mask_token_id,
                                  dtype=torch.long, device=device)
        block_tokens[:, 0] = greedy_first.squeeze()

        ctx = target_features[:, -block_size:, :] if target_features.shape[1] >= block_size \
            else F.pad(target_features, (0, 0, block_size - target_features.shape[1], 0))

        ctx_positions = torch.arange(
            max(0, seq_len - block_size), seq_len, device=device,
        )
        if len(ctx_positions) < block_size:
            ctx_positions = F.pad(ctx_positions, (block_size - len(ctx_positions), 0))
        draft_positions = torch.arange(seq_len, seq_len + block_size, device=device)
        position_ids = torch.cat([ctx_positions, draft_positions]).unsqueeze(0)

        noise_emb = embed_fn(block_tokens)

        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            draft_hidden = draft_model(
                noise_embedding=noise_emb,
                target_hidden=ctx[:, :block_size, :],
                position_ids=position_ids,
            )
            draft_logits = lm_head(draft_hidden[:, 1:, :])
            draft_tokens = torch.argmax(draft_logits, dim=-1)

        block_tokens[:, 1:] = draft_tokens

        full_input = torch.cat([prompt_ids, block_tokens], dim=1)
        with torch.amp.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda"):
            verify_out = target_model(full_input, use_cache=False)

        verify_logits = verify_out.logits[:, -(block_size):, :]
        verify_tokens = torch.argmax(verify_logits, dim=-1)

        for k in range(block_size - 1):
            per_pos_total[k] += 1
            if block_tokens[0, k + 1] == verify_tokens[0, k]:
                per_pos_correct[k] += 1

        total_blocks += 1

    per_pos_acc = (per_pos_correct / per_pos_total.clamp(min=1)).tolist()

    accepted_sim = 0
    for acc in per_pos_acc:
        if acc > 0.5:
            accepted_sim += acc
        else:
            break

    return {
        "tier": 1,
        "per_position_accuracy": per_pos_acc,
        "mean_position_accuracy": sum(per_pos_acc) / max(len(per_pos_acc), 1),
        "estimated_acceptance": accepted_sim + 1,
        "total_blocks": total_blocks,
    }


@torch.inference_mode()
def tier2_eval(
    target_model, draft_model, eval_data, block_size, mask_token_id,
    device, dtype, max_new_tokens=128,
):
    """
    Tier 2: Full speculative decoding evaluation.
    Runs actual speculative decoding and measures acceptance length.
    """
    result = evaluate_acceptance_length(
        target_model, draft_model, eval_data,
        block_size, mask_token_id,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    result["tier"] = 2
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(CACHE_DIR / "draft_checkpoint_final.pt"))
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2])
    parser.add_argument("--max-prompts", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading checkpoint: {args.checkpoint}")
    draft_model, config = load_draft_model(args.checkpoint, device=str(device), dtype=dtype)
    print(f"Draft model: {config.num_draft_layers} layers, block_size={config.block_size}")

    print(f"Loading target model: {TARGET_MODEL}")
    target_model, tokenizer = load_target_model(TARGET_MODEL, device=str(device), dtype=dtype)

    print(f"Loading eval prompts from {EVAL_DATA_PATH}")
    eval_data_raw = torch.load(EVAL_DATA_PATH, weights_only=True)
    eval_prompts = eval_data_raw["input_ids"][:args.max_prompts]
    print(f"Evaluating on {len(eval_prompts)} prompts")

    mask_token_id = 0

    print(f"\nRunning Tier {args.tier} evaluation...")
    t0 = time.time()

    if args.tier == 1:
        results = tier1_eval(
            target_model, draft_model, eval_prompts,
            config.block_size, mask_token_id, device, dtype,
        )
    else:
        results = tier2_eval(
            target_model, draft_model, eval_prompts,
            config.block_size, mask_token_id, device, dtype,
        )

    eval_time = time.time() - t0
    results["eval_seconds"] = eval_time
    results["num_prompts"] = len(eval_prompts)
    results["block_size"] = config.block_size
    results["num_draft_layers"] = config.num_draft_layers

    print()
    print("=" * 60)
    print(f"Tier {args.tier} Evaluation Results")
    print("=" * 60)

    if args.tier == 1:
        print(f"mean_position_accuracy: {results['mean_position_accuracy']:.4f}")
        print(f"estimated_acceptance:   {results['estimated_acceptance']:.2f}")
        print(f"per_position_accuracy:  {[f'{a:.3f}' for a in results['per_position_accuracy']]}")
    else:
        print(f"mean_accepted_length:   {results['mean_accepted_length']:.2f}")
        print(f"per_position_accuracy:  {[f'{a:.3f}' for a in results['per_position_accuracy']]}")
        print(f"total_blocks:           {results['total_blocks']}")

    print(f"eval_seconds:           {eval_time:.1f}")

    score = results.get("mean_accepted_length",
                        results.get("estimated_acceptance", 0))
    results["score"] = score
    print(f"\nscore: {score:.4f}")

    results_path = CACHE_DIR / "eval_results.json"

    serializable = {}
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
