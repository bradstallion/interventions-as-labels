"""Chapter 1, snippet 4: render a message list through the model's own chat
template. The message list is the one from snippet 3. CPU only; downloads the
Qwen tokenizer (a few MB) from Hugging Face on first use.
"""
import json
from pathlib import Path

from transformers import AutoTokenizer

RECORDS = Path("store/derived/m2_collect_qwen3B_bf16_conv/records.jsonl")
with RECORDS.open() as fh:
    messages = json.loads(fh.readline())["messages_per_step"][0]

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(tok.apply_chat_template(messages, tokenize=False,
                              add_generation_prompt=True))
