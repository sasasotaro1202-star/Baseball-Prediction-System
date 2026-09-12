"""Generate research evaluation reports for baseball models."""

from datetime import datetime


def generate_report(metrics, output_path="evaluation_report.json"):
    import json

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": metrics
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report
