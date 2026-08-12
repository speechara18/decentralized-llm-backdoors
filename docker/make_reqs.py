import sys
REMOVE_PREFIXES = ("torch==", "torch ", "torch=", "torchvision", "torchaudio",
                   "flash_attn", "flash-attn", "auto_gptq", "auto-gptq")
kept, dropped = [], []
for line in sys.stdin:
    s = line.strip()
    if not s or s.startswith("#"):
        continue
    if any(s.lower().startswith(p) for p in REMOVE_PREFIXES):
        dropped.append(s); continue
    # Relax exact pins to >= so a yanked exact version doesn't break the whole install
    s = s.replace("==", ">=")
    kept.append(s)
sys.stderr.write("DROPPED:\n  " + "\n  ".join(dropped) + "\n")
sys.stdout.write("\n".join(kept) + "\n")
