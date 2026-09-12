"""Baseball research weakness analyzer.

Analyzes evaluation results and generates next research priorities.
"""


def detect_weakness(metrics, threshold=0.0):
    weaknesses = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value < threshold:
            weaknesses.append(name)
    return weaknesses


def suggest_improvements(weaknesses):
    suggestions = []
    mapping = {
        "win_accuracy": "test starting pitcher and bullpen features",
        "score_mae": "improve scoring distribution model",
        "low_high_accuracy": "optimize total run features",
    }
    for item in weaknesses:
        suggestions.append(mapping.get(item, f"investigate {item}"))
    return suggestions


if __name__ == "__main__":
    print("Baseball research analyzer ready")
