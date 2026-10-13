"""Broader follow-up to fix_cyrillic_glitch.py: that fix only searched for
Cyrillic (the one script we happened to catch in v4's output). Then a live
v5 test turned up a Thai-script glitch too, proving other scripts can leak
in the same way. This scans every task title in the rich task bank for ANY
character outside Arabic, Latin, digits, and common punctuation, not just
Cyrillic, so we catch the whole class of bug instead of one instance of it.
"""
import json
import re

PATH = "datasets/rich_task_banks.json"
ALLOWED_RE = re.compile(
    r"[^\u0600-\u06FF\u0750-\u077Fa-zA-Z0-9\s.,!?()\-_/:;'\"\u060C\u061B\u061F%#@&+=]"
)

with open(PATH, encoding="utf-8") as f:
    banks = json.load(f)

flagged = []
for key, variations in banks.items():
    for variation in variations:
        for task in variation:
            matches = ALLOWED_RE.findall(task["title"])
            if matches:
                flagged.append({"key": key, "title": task["title"], "foreign_chars": list(set(matches))})

out = {"count": len(flagged), "items": flagged}
print(json.dumps(out, ensure_ascii=False, indent=2))
