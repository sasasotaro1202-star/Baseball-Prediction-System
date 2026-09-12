"""Baseball evaluation condition analysis.

Analyzes prediction performance by available categorical conditions.
Designed as part of the autonomous research pipeline.
"""

from collections import defaultdict


def analyze_by_condition(records, condition_key, correct_key="correct"):
    groups = defaultdict(lambda: {"total": 0, "correct": 0})

    for row in records:
        condition = row.get(condition_key, "unknown")
        groups[condition]["total"] += 1
        if row.get(correct_key):
            groups[condition]["correct"] += 1

    result = {}
    for key, value in groups.items():
        total = value["total"]
        result[key] = {
            "samples": total,
            "accuracy": value["correct"] / total if total else 0
        }

    return result


def find_weak_conditions(records, condition_keys):
    weaknesses = {}
    for key in condition_keys:
        analysis = analyze_by_condition(records, key)
        weaknesses[key] = sorted(
            analysis.items(),
            key=lambda x: x[1]["accuracy"]
        )
    return weaknesses
