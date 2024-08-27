from __future__ import annotations
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.metrics import roc_auc_score


np.random.seed(0)


def generate_label_scores(n_samples: int = 1000, pos_rate: float = 0.5):
    z = np.random.normal(size=n_samples)
    score = sigmoid(z)
    threshold = np.quantile(score, pos_rate)
    y = np.where(score < threshold, 0, 1)
    flip = np.random.rand(n_samples) < 0.1
    y[flip] = 1 - y[flip] 
    return y, score


def auc_score_1(y_true: list[float], y_score: list[float]) -> float:
    """
    https://github.com/microsoft/LightGBM/blob/v1/src/metric/binary_metric.hpp#L181
    """
    # sort labels by score in descending order
    label_and_score = sorted(zip(y_true, y_score), key=lambda s: -s[1])
    
    n_pos, n_neg = 0, 0
    cur_pos, cur_neg = 0, 0
    threshold = label_and_score[0][1]
    num = 0
    i = 0
    for label, score in label_and_score:
        if score < threshold:
            num += cur_neg * (n_pos + 0.5 * cur_pos)  # 0.5 * cur_neg * cur_pos corresponds to ties   
            n_pos += cur_pos
            n_neg += cur_neg
            cur_pos = cur_neg = 0
            threshold = score
        
        cur_pos += label
        cur_neg += 1 - label
        i += 1
    num += cur_neg * (n_pos + 0.5 * cur_pos)  # 0.5 * cur_neg * cur_pos corresponds to ties   
    n_pos += cur_pos
    n_neg += cur_neg
            
    return num / (n_pos * n_neg)


def auc_by_sort(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_pos, n_neg = 0, 0
    count = 0
    values = sorted(zip(y_true, y_pred), key=lambda t: t[1])
    for label, _ in values:
        if label == 1:
            count += n_neg
            n_pos += 1
        else:
            n_neg += 1
    return count / (n_pos * n_neg)


y_true, y_score = generate_label_scores(n_samples=10000, pos_rate=0.3)
y_score = np.round(y_score, 2)
y_true = y_true.tolist()
y_score = y_score.tolist()
print(f'postive rate: {np.mean(y_true)}, {np.mean(y_score)}')
auc_ref = roc_auc_score(y_true, y_score)
auc_1 = auc_score_1(y_true, y_score)
auc_2 = auc_by_sort(y_true, y_score)

print(f'auc_ref = {auc_ref}, {auc_1 = }, {auc_2 = }')