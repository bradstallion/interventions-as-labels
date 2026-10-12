"""Prompt rendering for hooked capture + the token-level replay invariant.

This is the rollout-to-measurement boundary (paper §4.1): activations must be
captured on EXACTLY the token sequence the agent was conditioned on during
collection. Two rendering modes:

- "plain": system folded into the text, no chat template. For non-chat dev
  models (gpt2/pythia) — matches nothing collected, dev-tier only.
- "chat": the model's chat template via the SAME code path as HFBackend
  (including the Gemma system-role fold). For gate runs.

verify_token_replay() is the blocking check: HF tokenization of the rendered
prompt must equal the tokens the hooked (TransformerLens) model will run on.
"""

from __future__ import annotations

from ige.agent.backend import merge_system_into_user

Messages = list[dict[str, str]]


def render_plain(messages: Messages) -> str:
    """System folded into the first user turn, turns joined by blank lines."""
    merged = merge_system_into_user(messages)
    return "\n\n".join(m["content"] for m in merged)


def render_chat(tokenizer, messages: Messages) -> str:
    """Render with the model's chat template (generation prompt included),
    falling back to the system fold exactly like HFBackend.generate."""
    from jinja2.exceptions import TemplateError

    def _render(msgs: Messages) -> str:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    try:
        return _render(messages)
    except TemplateError:
        return _render(merge_system_into_user(messages))


class TokenReplayError(AssertionError):
    pass


def verify_token_replay(hf_tokenizer, tl_model, text: str) -> list[int]:
    """Blocking invariant: TL must run on the same token ids HF produces.

    Chat templates already include BOS; render_plain does not, and TL's
    to_tokens prepends BOS by default — so we compare against TL tokenization
    with prepend_bos=False and require the HF ids to match exactly.
    Returns the verified token ids.
    """
    hf_ids = hf_tokenizer(text, add_special_tokens=False)["input_ids"]
    tl_ids = tl_model.to_tokens(text, prepend_bos=False)[0].tolist()
    if hf_ids != tl_ids:
        first = next((i for i, (a, b) in enumerate(zip(hf_ids, tl_ids)) if a != b),
                     min(len(hf_ids), len(tl_ids)))
        raise TokenReplayError(
            f"token replay drift at index {first}: HF {hf_ids[first:first+4]} vs "
            f"TL {tl_ids[first:first+4]} (len {len(hf_ids)} vs {len(tl_ids)}) — "
            "activations would be captured on tokens the agent never saw"
        )
    return hf_ids


def replay_tokens_for_record(hf_tokenizer, tl_model, record: dict,
                             step: int = 0) -> list[int]:
    """Render + verify a record's step for hooked re-execution.

    Two blocking invariants, in order:
    1. verify_token_replay: HF and TL tokenize the re-rendered text identically.
    2. If the record carries rollout provenance (`rollout_info`, written by
       HFBackend at collection time), the re-rendered ids must equal the ids
       the agent ACTUALLY ran on at rollout — this is the rollout->replay
       identity that (1) alone cannot guarantee (a changed chat template,
       tokenizer or model revision would pass (1) and silently violate it;
       external adversarial audit, 2026-08-06).

    Records collected before rollout_info existed fall back to (1) only —
    callers get the legacy guarantee, not the stronger one.
    """
    text = render_chat(hf_tokenizer, record["messages_per_step"][step])
    ids = verify_token_replay(hf_tokenizer, tl_model, text)
    infos = record.get("rollout_info") or []
    info = infos[step] if step < len(infos) else None
    if info and info.get("prompt_token_ids"):
        want = list(info["prompt_token_ids"])
        if ids != want:
            first = next((i for i, (a, b) in enumerate(zip(ids, want)) if a != b),
                         min(len(ids), len(want)))
            raise TokenReplayError(
                f"rollout->replay drift at index {first}: replay {ids[first:first+4]} "
                f"vs rollout {want[first:first+4]} (len {len(ids)} vs {len(want)}; "
                f"rollout tokenizer {info.get('tokenizer')}, "
                f"model revision {info.get('model_revision')}) — attribution "
                "would run on tokens the agent never saw")
    return ids


def instruction_final_index(tokenizer, text: str) -> int:
    """Token index of the last instruction token: the token immediately before
    the workspace block. Positions up to here are BIT-IDENTICAL between the
    twins of a pair (same prefix + causal attention) — the positional canary.
    """
    marker = "\nWorkspace items:"
    char_idx = text.index(marker)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    last = max(i for i, (start, _end) in enumerate(enc["offset_mapping"]) if start < char_idx
               and _end <= char_idx)
    return last
