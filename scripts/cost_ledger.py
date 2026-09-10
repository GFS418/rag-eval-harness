"""Sum spend recorded in the LLM response cache (the single source of truth
for every call this project has made), by model. With --cap, exit 1 if the
total exceeds it, so a pipeline can stop before overspending.
Batch-submitted calls are stored at list price; pass --batch-discount 0.5
to report what they actually cost."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm import CACHE_PATH, price_for  # noqa: E402


def totals(path: Path = CACHE_PATH) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    conn = sqlite3.connect(path)
    for (blob,) in conn.execute("SELECT response FROM responses"):
        r = json.loads(blob)
        m = r.get("model_id") or "unknown"
        d = out.setdefault(m, {"n": 0, "input_tokens": 0, "output_tokens": 0, "usd_list": 0.0, "n_batch": 0})
        d["n"] += 1
        d["input_tokens"] += r["input_tokens"]
        d["output_tokens"] += r["output_tokens"]
        pi, po = price_for(m)
        d["usd_list"] += (r["input_tokens"] * pi + r["output_tokens"] * po) / 1e6
        if r.get("latency_s") is None:
            d["n_batch"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--batch-discount", type=float, default=0.5)
    args = ap.parse_args()
    t = totals()
    total = 0.0
    for m, d in sorted(t.items()):
        frac_batch = d["n_batch"] / d["n"] if d["n"] else 0
        est = d["usd_list"] * (1 - frac_batch * (1 - args.batch_discount))
        total += est
        print(f"{m:22s} calls={d['n']:5d} (batch {d['n_batch']:5d})  in={d['input_tokens']:>9,}  out={d['output_tokens']:>8,}  "
              f"list=${d['usd_list']:.2f}  est=${est:.2f}")
    print(f"TOTAL est ${total:.2f}" + (f"  (cap ${args.cap:.2f})" if args.cap else ""))
    if args.cap is not None and total > args.cap:
        print("OVER CAP: stopping")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
