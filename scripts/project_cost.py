"""Project the overnight spend from the smoke test's measured token usage and
refuse to continue if it exceeds the cap. Output tokens include thinking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm import PRICE_PER_MTOK  # noqa: E402

BATCH = 0.5


def usd(model: str, n: int, tin: float, tout: float) -> float:
    pi, po = PRICE_PER_MTOK[model]
    return n * (tin * pi + tout * po) / 1e6 * BATCH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--already-spent", type=float, default=0.0)
    args = ap.parse_args()
    df = pd.read_parquet(args.smoke)
    tin, tout = df["input_tokens"].mean(), df["output_tokens"].mean()
    n = args.n
    gen = {
        "A rag ft (sonnet)": usd("claude-sonnet-5", n, tin, tout),
        "B rag base (sonnet)": usd("claude-sonnet-5", n, tin, tout),
        "D rag ft (haiku)": usd("claude-haiku-4-5", n, tin, tout),
        "E closed-book (sonnet)": usd("claude-sonnet-5", n, 250, tout),
        "F oracle (sonnet)": usd("claude-sonnet-5", n, tin * 0.6, tout),
    }
    n_ans = int(n * 0.85)   # ~15% abstain
    judge = {  # Opus at low effort, verdict-only schema
        "correctness judge x5 arms (opus)": usd("claude-opus-5", 5 * n_ans, 450, 120),
        "faithfulness judge, primary arm (opus)": usd("claude-opus-5", n_ans, tin + 400, 220),
    }
    optional = {
        "opt: faithfulness judge on base arm": usd("claude-opus-5", n_ans, tin + 400, 220),
        "opt: faithfulness judge on haiku arm": usd("claude-opus-5", n_ans, tin + 400, 220),
        "opt: C rag fixed-512 gen + correctness": usd("claude-sonnet-5", n, tin * 2.6, tout) + usd("claude-opus-5", n_ans, 450, 120),
    }
    total = sum(gen.values()) + sum(judge.values()) + args.already_spent
    print(f"smoke: n={len(df)} mean input={tin:.0f} mean output={tout:.0f} tokens (output includes thinking)")
    for k, v in {**gen, **judge}.items():
        print(f"  {k:38s} ${v:5.2f}")
    print(f"  already spent                          ${args.already_spent:5.2f}")
    print(f"PROJECTED TOTAL (core) ${total:.2f}  (cap ${args.cap:.2f})")
    for k, v in optional.items():
        print(f"  {k:38s} ${v:5.2f}  (runs only if still under cap)")
    if total > args.cap:
        print("PROJECTION OVER CAP: not submitting the full runs")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
