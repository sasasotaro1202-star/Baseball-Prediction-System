"""Creates next research tasks for baseball model improvement."""


def create_plan(analysis):
    return {
        "current_issue": analysis,
        "next_steps": [
            "run backtest",
            "compare candidate model",
            "evaluate out of sample performance",
            "save research history"
        ]
    }
