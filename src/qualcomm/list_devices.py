"""
List every device available in the Qualcomm AI Hub device farm.

Run on either machine, in .venv-qai, after `qai-hub configure`.

Device name strings are exact and case-sensitive. Copy them from THIS output
into configs/experiment.yaml — never type them from memory or from a blog post.
"""

import sys

try:
    import qai_hub as hub
except ImportError:
    sys.exit("qai_hub not installed. Are you in .venv-qai?")

try:
    devices = hub.get_devices()
except Exception as e:
    sys.exit(
        f"Could not reach AI Hub: {e}\n\n"
        "Check your token:  qai-hub configure --api_token <TOKEN>\n"
        "Get one at: workbench.aihub.qualcomm.com -> Account -> Settings"
    )

print(f"{len(devices)} devices available\n")

INTERESTING = ("9075", "8550", "X Elite", "8 Gen 3", "8275", "RB3", "Vision Kit")

print("--- likely relevant to this project ---")
for d in devices:
    if any(k.lower() in d.name.lower() for k in INTERESTING):
        print(f"  {d.name}")

print("\n--- all devices ---")
for d in devices:
    print(f"  {d.name}")
