from constants import NO_OF_FEATURES
from sklearn.metrics import r2_score


def r2_score_adjusted(y_true, y_pred):
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    n = len(y_pred)
    den = n - NO_OF_FEATURES - 1
    num = (n-1) * (1-r2)
    return 1 - (num/den)
