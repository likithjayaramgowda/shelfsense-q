"""
Machine handoff helper. Enforces the relay rule.

    python scripts/handoff.py start    # pull, then show HANDOFF.md
    python scripts/handoff.py stop     # check secrets, commit, push
"""

import subprocess
import sys
from pathlib import Path


def run(cmd, check=True):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def start():
    run(["git", "pull", "--rebase"])
    print("\n" + "=" * 60)
    print(Path("HANDOFF.md").read_text(encoding="utf-8"))
    print("=" * 60)


def stop():
    # Guard: never let .env reach the index.
    staged = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout
    if any(".env" in line and ".env.example" not in line for line in staged.splitlines()):
        sys.exit(
            "REFUSING TO COMMIT: .env appears in git status.\n"
            "It must be ignored. Check .gitignore, then run:\n"
            "  git rm --cached .env"
        )

    print("Did you update HANDOFF.md? (last machine / done / next / blockers)")
    if input("y to continue: ").strip().lower() != "y":
        sys.exit("Update HANDOFF.md first.")

    msg = input("Commit message: ").strip() or "wip"
    run(["git", "add", "-A"])
    run(["git", "commit", "-m", msg], check=False)
    run(["git", "push"])
    print("\nPushed. Safe to switch machines.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("start", "stop"):
        sys.exit(__doc__)
    (start if sys.argv[1] == "start" else stop)()
