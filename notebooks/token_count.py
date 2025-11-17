import os
import tiktoken
from pathlib import Path

encoding = tiktoken.get_encoding("cl100k_base")

dirs = [Path("001"), Path("002")]

def count_tokens_in_dir(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            full = os.path.join(root, f)
            try:
                with open(full, "r", encoding="utf8", errors="ignore") as h:
                    text = h.read()
                    total += len(encoding.encode(text))
            except:
                pass
    return total

for d in dirs:
    print(d, count_tokens_in_dir(d))
