"""M5: LoRA fine-tune from overseer interventions (criteria written down in advance: experiments/m5_edit).

Two variants share this runner and ALL hyperparameters; the only difference
is `layers_to_transform`:
- local: layers around the causal site (the mechanistic edit)
- bc:    all layers (the interactive-imitation-learning baseline)

Training data comes from the collection records, TRAIN pairs only:
- intervened episodes (overask / proceed_when_ambiguous): target = the
  overseer's canonical `correction` — the edit is grounded in interventions,
  never in ground truth;
- clean episodes (correct_clarify / correct_act): target = the model's own
  first reply (retention: keep asking when ambiguous).

Loss on reply tokens only (prompt tokens masked at -100)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ige.attribution.probes import split_by_pair
from ige.attribution.render import render_chat
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed


def build_dataset(records: list[dict], *, split_seed: int = 0) -> list[dict]:
    """One example per TRAIN-pair record: {'messages', 'target', 'kind'}.
    Corrections use verdict.correction; retention uses the model's own reply."""
    pair_ids = [r["pair_id"] for r in records]
    train_idx, _ = split_by_pair(pair_ids, seed=split_seed)
    examples = []
    for i in train_idx:
        r = records[i]
        if r["verdict"]["intervene"]:
            target = r["verdict"]["correction"]
            kind = "correction"
            if not target:
                continue  # an intervention without canonical correction is unusable
        else:
            target = r["raw_replies"][0].strip()
            kind = "retention"
        examples.append({"messages": r["messages_per_step"][0],
                         "target": target, "kind": kind})
    return examples


def encode_example(tokenizer, ex: dict, max_len: int = 512):
    """input_ids + labels with the prompt masked; target ends with newline
    (the episode-runner's stop signal)."""
    prompt = render_chat(tokenizer, ex["messages"])
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(ex["target"] + "\n", add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + t_ids)[:max_len]
    labels = ([-100] * len(p_ids) + t_ids)[:max_len]
    return input_ids, labels


def run(cfg) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "bfloat16")))

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    examples = build_dataset(records, split_seed=int(cfg.get("split_seed", 0)))
    n_corr = sum(e["kind"] == "correction" for e in examples)
    print(f"dataset: {len(examples)} examples ({n_corr} corrections, "
          f"{len(examples) - n_corr} retention)")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=dtype)
    layers = cfg.get("layers_to_transform", None)
    lora = LoraConfig(
        r=int(cfg.get("lora_r", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=0.05,
        target_modules=list(cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])),
        layers_to_transform=list(layers) if layers is not None else None,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device).train()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.get("lr", 1e-4)))
    epochs = int(cfg.get("epochs", 2))
    accum = int(cfg.get("grad_accum", 8))

    step = 0
    losses: list[float] = []
    for epoch in range(epochs):
        order = torch.randperm(len(examples)).tolist()
        opt.zero_grad()
        for j, idx in enumerate(order):
            input_ids, labels = encode_example(tokenizer, examples[idx])
            out = model(
                input_ids=torch.tensor([input_ids], device=device),
                labels=torch.tensor([labels], device=device))
            (out.loss / accum).backward()
            losses.append(float(out.loss))
            if (j + 1) % accum == 0:
                opt.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    avg = sum(losses[-10 * accum:]) / len(losses[-10 * accum:])
                    print(f"epoch {epoch} step {step} loss {avg:.4f}")
        opt.step()
        opt.zero_grad()

    out_dir = Path(cfg.get("out", "store/derived/m5_lora"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir / "adapter"))
    report = {
        "n_examples": len(examples), "n_corrections": n_corr,
        "trainable_params": trainable,
        "final_loss": sum(losses[-50:]) / max(1, len(losses[-50:])),
        "layers_to_transform": list(layers) if layers is not None else "all",
        "meta": run_metadata(dict(lora_r=int(cfg.get("lora_r", 8)), epochs=epochs)),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "meta"}))
    return report
