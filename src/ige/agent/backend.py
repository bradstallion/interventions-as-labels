"""LLM backends behind one interface.

- FakeBackend: scripted policies for unit tests (deterministic, no deps).
- HFBackend: transformers generation, greedy, optional 4-bit — the
  dev-prototyping tier (configs/model/gemma2_2b_4bit.yaml). Generation only:
  interpretability goes through ige.hooks, never through here.
"""

from __future__ import annotations

from typing import Callable, Protocol

Messages = list[dict[str, str]]


class LLMBackend(Protocol):
    name: str

    def generate(self, messages: Messages, max_new_tokens: int = 40) -> str: ...


class FakeBackend:
    """Replies via a policy function(messages) -> str."""

    def __init__(self, policy: Callable[[Messages], str], name: str = "fake"):
        self.name = name
        self._policy = policy

    def generate(self, messages: Messages, max_new_tokens: int = 40) -> str:
        return self._policy(messages)


def merge_system_into_user(messages: Messages) -> Messages:
    """Fold a leading system message into the first user turn.

    Some chat templates (e.g. Gemma) reject the system role; the content is
    preserved verbatim at the top of the first user message. Traces store the
    role-structured message list, so the replay invariant is unaffected.
    """
    if not messages or messages[0]["role"] != "system":
        return messages
    system, rest = messages[0], list(messages[1:])
    if not rest or rest[0]["role"] != "user":
        return [{"role": "user", "content": system["content"]}, *rest]
    rest[0] = {"role": "user", "content": f"{system['content']}\n\n{rest[0]['content']}"}
    return rest


class HFBackend:
    """Greedy chat generation via transformers; supports 4-bit on 6GB GPUs."""

    def __init__(self, model_name: str, *, quantization: str | None = None, device: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"{model_name}{'-4bit' if quantization == '4bit' else ''}"
        kwargs: dict = {"dtype": torch.bfloat16}
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device, **kwargs
        )
        self.model.eval()

    def generate(self, messages: Messages, max_new_tokens: int = 40) -> str:
        import torch
        from jinja2.exceptions import TemplateError

        def render_text(msgs: Messages) -> str:
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)

        used = messages
        try:
            text = render_text(used)
        except TemplateError:  # e.g. Gemma: "System role not supported"
            used = merge_system_into_user(messages)
            text = render_text(used)
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        # rollout-side invariant: the text render must retokenize to the ids
        # we actually run on, or the persisted provenance would be fiction
        direct = self.tokenizer.apply_chat_template(
            used, add_generation_prompt=True, tokenize=True)
        if hasattr(direct, "keys"):  # transformers >= 5 returns a BatchEncoding
            direct = direct["input_ids"]
        prompt_ids = enc["input_ids"][0].tolist()
        assert prompt_ids == list(direct), (
            "chat-template text render and tokenized render disagree — "
            "rollout provenance would not describe the executed prompt")
        inputs = enc.to(self.model.device)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # rollout provenance for the replay boundary: everything
        # a later re-execution must reproduce bit-identically
        self.last_info = {
            "rendered_text": text,
            "prompt_token_ids": prompt_ids,
            "first_reply_token_id": int(out[0, n_prompt]) if out.shape[1] > n_prompt else None,
            "tokenizer": getattr(self.tokenizer, "name_or_path", None),
            "model_revision": getattr(self.model.config, "_commit_hash", None),
        }
        return self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
