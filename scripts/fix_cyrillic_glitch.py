"""One-off cleanup: the rich task bank generator occasionally hallucinated
mixed Arabic/Cyrillic glitch words (e.g. Arabic "ت" + Cyrillic "ест" instead
of "اختبار") in datasets/rich_task_banks.json. v4 learned this exact glitch
from those bad labels. This scrubs it before v5 trains on the same file.
"""
import json
import re

PATH = "datasets/rich_task_banks.json"

REPLACEMENTS = [
    ("ل управл", "لإدارة"),  # "ل управл" -> "لإدارة"
    ("управл", "إدارة"),  # "управл" -> "إدارة"
    ("базة", "قاعدة"),  # "базة" -> "قاعدة"
    ("модель", "نموذج"),  # "модель" -> "نموذج"
    ("تест", "اختبار"),  # Arabic ت + Cyrillic "ест" -> "اختبار"
]

CYRILLIC_RE = re.compile("[Ѐ-ӿ]")


def main():
    with open(PATH, encoding="utf-8") as f:
        banks = json.load(f)

    fixed_count = 0
    for variations in banks.values():
        for variation in variations:
            for task in variation:
                original = task["title"]
                new_title = original
                for bad, good in REPLACEMENTS:
                    new_title = new_title.replace(bad, good)
                if new_title != original:
                    fixed_count += 1
                    task["title"] = new_title

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(banks, f, ensure_ascii=False, indent=2)

    remaining = sum(
        1
        for variations in banks.values()
        for variation in variations
        for task in variation
        if CYRILLIC_RE.search(task["title"])
    )
    print(f"fixed_count={fixed_count} remaining_cyrillic={remaining}")


if __name__ == "__main__":
    main()
