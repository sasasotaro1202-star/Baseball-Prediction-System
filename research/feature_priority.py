"""Feature research priority planner for baseball prediction."""

FEATURE_AREAS = [
    "starting_pitcher",
    "bullpen_fatigue",
    "batting_form",
    "park_factor",
    "weather",
    "schedule_fatigue",
]


def rank_features(weaknesses):
    priorities = []
    for feature in FEATURE_AREAS:
        score = 1
        if feature.replace("_", "") in "".join(weaknesses).replace("_", ""):
            score += 10
        priorities.append({"feature": feature, "priority": score})
    return sorted(priorities, key=lambda x: x["priority"], reverse=True)
