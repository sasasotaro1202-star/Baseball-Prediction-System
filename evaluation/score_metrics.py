"""Baseball score prediction evaluation metrics."""


def mae(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(abs(a-b) for a,b in zip(y_true,y_pred)) / len(y_true)


def rmse(y_true, y_pred):
    if not y_true:
        return 0.0
    return (sum((a-b)**2 for a,b in zip(y_true,y_pred)) / len(y_true)) ** 0.5


def exact_rate(y_true, y_pred):
    if not y_true:
        return 0.0
    return sum(a == b for a,b in zip(y_true,y_pred)) / len(y_true)
