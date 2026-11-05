"""Quick check of the GSM8K extraction and grading in src/gsm8k_check.py (run in the WSL venv)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from gsm8k_check import grade, last_boxed, tail_dup8  # noqa: E402

cases = [("so \\boxed{18}.", "18", True), ("\\boxed{18.0}", "18", True), ("\\boxed{\\$18}", "18", True),
         ("\\boxed{1,000}", "1000", True), ("\\boxed{\\frac{1}{2}}", "0.5", True), ("\\boxed{17}", "18", False),
         ("no box here", "18", False), ("\\boxed{3} then \\boxed{18}", "18", True)]
bad = 0
for text, gold, want in cases:
    got = grade(last_boxed(text), gold)
    bad += got != want
    print(f"{'ok ' if got == want else 'BAD'} {text!r:35} gold={gold:5} -> {got}")
print("tail_dup8 loop:", round(tail_dup8("Wait, if the ratio is 1:2, then 1/2 of 2 is 1. " * 50), 2),
      "| prose:", round(tail_dup8(" ".join(f"w{i}" for i in range(300))), 2))
sys.exit(1 if bad else 0)
