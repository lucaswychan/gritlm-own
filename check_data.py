#!/usr/bin/env python
"""
Quick data sanity-check for the hard-neg datasets.

• prints how many examples have <3 negatives (your original check)
• flags queries whose text part is empty
• flags queries whose text part is lost after the 512-token truncation
  – these are the ones that make attention_mask become all-zeros later.
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------
directory   = "/data/wychanbu/re_data/hard-neg/"   # folder with the *.jsonl files
model_name  = "TIGER-Lab/General-Reasoner-Qwen2.5-7B"
query_max_len = 1024                                 # must match the value used in training
# ---------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data/wychanbu/huggingface/hub")

# Special-token strings – use the same as in training.run / data.py
BASE_BOS    = ""
USER_BOS    = ""
USER_EOS    = ""
EMBED_BOS   = ""
EMBED_EOS   = ""

def tokens_after_instruction(instr: str, text: str) -> int:
    """How many tokens remain once the instruction part is masked out?"""
    prompt_full  = BASE_BOS + USER_BOS + instr + USER_EOS + EMBED_BOS + text + EMBED_EOS
    prompt_instr = BASE_BOS + USER_BOS + instr + USER_EOS + EMBED_BOS
    tok_full  = tokenizer(prompt_full,  add_special_tokens=False,
                          truncation=True, max_length=query_max_len)["input_ids"]
    tok_instr = tokenizer(prompt_instr, add_special_tokens=False)["input_ids"]
    return len(tok_full) - len(tok_instr)

for file in sorted(os.listdir(directory)):
    if not file.endswith(".jsonl"):
        continue

    path     = os.path.join(directory, file)
    dataset  = load_dataset("json", data_files=path, split="train", cache_dir=directory)

    total_len   = len(dataset)
    neg_lt_3    = 0
    empty_text  = 0
    instr_only  = 0

    for ex in dataset:
        if len(ex["neg"]) < 3:
            neg_lt_3 += 1

        # query can be string or [instruction, text] tuple/list
        q_raw = ex["query"]
        if isinstance(q_raw, str):           # no separate instruction
            instr, text = "", q_raw
        else:
            instr, text = q_raw[0], q_raw[1]

        if not text.strip():
            empty_text += 1
            continue                         # no need to check further

        if tokens_after_instruction(instr, text) == 0:
            instr_only += 1

    print(f"File: {file}")
    print(f"  Total examples           : {total_len}")
    print(f"  Neg < 3                  : {neg_lt_3}  ({neg_lt_3/total_len*100:5.2f}%)")
    print(f"  Query text empty         : {empty_text}  ({empty_text/total_len*100:5.2f}%)")
    print(f"  Query lost after truncate: {instr_only}  ({instr_only/total_len*100:5.2f}%)")
    print("-" * 70)