"""Install the Anthropic API key for this project without it ever being
printed, pasted into a shell command, or logged.

    python scripts/set_api_key.py

Prompts for the key with hidden input, validates its shape, writes `.env`
(for the scripts) and `.streamlit/secrets.toml` (for the app), both
gitignored and chmod 600, then verifies it with one free API call.
Prints only OK / FAIL messages, never the key or any part of it.
"""
from __future__ import annotations

import getpass
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"^sk-ant-[A-Za-z0-9_\-]{40,}$")


def main() -> int:
    key = getpass.getpass("Paste the API key (input is hidden), then press Enter: ").strip()
    if not PATTERN.match(key):
        print("FAIL: that does not look like a single Anthropic API key "
              f"(got {len(key)} characters; expected one token starting with sk-ant-). Nothing written.")
        return 1
    env = ROOT / ".env"
    secrets = ROOT / ".streamlit" / "secrets.toml"
    secrets.parent.mkdir(exist_ok=True)
    env.write_text(f"ANTHROPIC_API_KEY={key}\n")
    secrets.write_text(f'ANTHROPIC_API_KEY = "{key}"\n')
    for p in (env, secrets):
        os.chmod(p, 0o600)
    del key
    print(f"wrote {env.relative_to(ROOT)} and {secrets.relative_to(ROOT)} (mode 600, gitignored)")

    sys.path.insert(0, str(ROOT / "src"))
    from llm import load_dotenv

    os.environ.pop("ANTHROPIC_API_KEY", None)
    load_dotenv(env)
    try:
        import anthropic

        anthropic.Anthropic().messages.count_tokens(
            model="claude-haiku-4-5", messages=[{"role": "user", "content": "ping"}])
    except anthropic.AuthenticationError:
        print("FAIL: the API rejected this key (authentication error). Files were written; re-run with the right key.")
        return 1
    except Exception as e:  # network etc.
        print(f"WARN: could not verify with the API ({type(e).__name__}); files were written.")
        return 0
    print("OK: key verified with the API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
