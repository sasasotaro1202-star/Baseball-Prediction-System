"""Baseball over/under evaluation metrics."""


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a,b in zip(y_true,y_pred)) / len(y_true)


def line_results(lines, actual_total, predicted):
    results = []
    for line, actual, pred in zip(lines, actual_total, predicted):
        results.append({"line": line, "actual": actual, "prediction": pred})
    return results
