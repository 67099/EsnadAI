"""
Uses a local Ollama model (qwen2.5:7b-instruct) as a one-time data-generation
assistant to write a REAL, full Work Breakdown Structure (WBS) for each
project — grounded in standard WBS practice (project management, design,
backend, frontend, QA, deployment — see the Atlassian WBS reference:
https://community.atlassian.com/forums/App-Central-articles/3-Work-Breakdown-Structure-WBS-Gantt-Chart-examples-for-software/ba-p/2338509)
and the 8/80 rule (each task is a specific, meaningful chunk of real work).

This is NOT the model being fine-tuned. ALLaM-7B-Instruct remains the only
model actually trained and shipped — qwen is a throwaway tool used once to
build better training *content*, exactly like a senior engineer/PM would
write a real project backlog, so the training data teaches genuine
whole-project task-breakdown skill instead of narrow per-feature snippets.

Output: datasets/rich_task_banks.json — one entry per project (2 full WBS
variations each), cached so this expensive step only needs to run once even
if we regenerate the final training data multiple times.
"""
import json
import os
import re
import sys
import time
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from generate_synthetic_data import PROJECTS  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "datasets", "rich_task_banks.json")

SKILL_CATEGORIES = ["frontend", "backend", "mobile", "ai", "devops", "qa", "design", "database", "none"]

PROMPT_TEMPLATE = """You are a senior project manager creating a complete Work Breakdown Structure (WBS) for a real student software project, following standard WBS practice (the 8/80 rule: each task is a specific, meaningful chunk of real work — never a vague one-line placeholder). Cover the project's FULL lifecycle, structured like this:

1. Project Management (requirements gathering, planning)
2. Design (UX/UI)
3. Backend Development
4. Frontend Development
5. Quality Assurance
6. Deployment

Project: {project_idea}
Key features it needs: {features}

Write between 14 and 20 tasks spanning ALL the phases above — not just development. Each task must be SPECIFIC to this exact project's real features — mention real technical details where relevant (actual data fields, specific validation rules, specific UI states, specific API behavior, specific edge cases, specific deployment steps). Do not write generic placeholders.

Categorize each task by which skill it needs: frontend, backend, mobile, ai, devops, qa, design, database, or none (use "none" for project-management/requirements/documentation tasks that don't need a specific specialist; use "devops" for deployment tasks; use "qa" for testing tasks; use "design" for UX/UI tasks).

Respond in Arabic. Output ONLY a JSON array, nothing else, matching exactly:
[{{"title": string, "difficulty": "easy"|"medium"|"hard", "skillCategory": string}}]
"""


def call_ollama(prompt, retries=3):
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.6},
    }).encode("utf-8")
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            return result["response"]
        except Exception as e:
            print(f"  retry {attempt+1}/{retries} after error: {e}")
            time.sleep(3)
    raise RuntimeError("Ollama call failed after retries")


def parse_tasks(raw_text):
    match = re.search(r"\[.*\]", raw_text, re.S)
    if not match:
        return None
    try:
        tasks = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    cleaned = []
    for t in tasks:
        if not isinstance(t, dict) or "title" not in t:
            continue
        skill = t.get("skillCategory", "none")
        if skill not in SKILL_CATEGORIES:
            skill = "none"
        difficulty = t.get("difficulty", "medium")
        if difficulty not in ("easy", "medium", "hard"):
            difficulty = "medium"
        cleaned.append({"title": t["title"], "difficulty": difficulty, "skillCategory": skill})
    return cleaned if cleaned else None


VARIATIONS_PER_PROJECT = 2


def main():
    banks = {}
    total = len(PROJECTS) * VARIATIONS_PER_PROJECT
    done = 0

    for project in PROJECTS:
        key = project["idea"]
        variations = []
        for v in range(VARIATIONS_PER_PROJECT):
            prompt = PROMPT_TEMPLATE.format(
                project_idea=project["idea"],
                features=", ".join(project["features"]),
            )
            if v > 0:
                prompt += "\nWrite a DIFFERENT valid breakdown than a typical one — vary which specific tasks and technical details you include."
            raw = call_ollama(prompt)
            tasks = parse_tasks(raw)
            if tasks is None:
                print(f"  FAILED to parse for: {key} (variation {v})\n  raw: {raw[:300]}")
                tasks = []
            variations.append(tasks)
            done += 1
            print(f"[{done}/{total}] {key} (v{v}) -> {len(tasks)} tasks")
        banks[key] = variations

        # Save after every project, not just at the end — a crash near the
        # finish line must not lose all prior work.
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(banks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(banks)} project WBS banks ({VARIATIONS_PER_PROJECT} variations each) to {OUTPUT_PATH}")

    empty = [k for k, variations in banks.items() for v in variations if not v]
    if empty:
        print(f"WARNING: {len(empty)} variations failed to parse and are empty")


if __name__ == "__main__":
    main()
