import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    d2_pinball_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)

from .basemetric import BaseMetric


def accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    as_count: bool = False,
) -> float | int:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    if as_count:
        return int(correct)
    return float(correct / max(targets.numel(), 1))


def _is_multiclass_logits(scores: torch.Tensor) -> bool:
    return scores.ndim > 1 and int(scores.size(-1)) > 1


def _flatten_targets(answers: torch.Tensor) -> np.ndarray:
    return answers.reshape(-1).numpy()


def _binary_probs_from_logits(scores: torch.Tensor) -> np.ndarray:
    return scores.sigmoid().reshape(-1).numpy()


class Acc1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))

        if _is_multiclass_logits(scores):  # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = _binary_probs_from_logits(scores)
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores > cutoff, 1, 0)
        return accuracy_score(answers, labels)

class Acc5(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).softmax(-1).numpy()
        answers = torch.cat(self.answers).numpy()
        num_classes = scores.shape[-1]
        k = min(5, num_classes)
        return top_k_accuracy_score(answers, scores, k=k, labels=np.arange(num_classes))

class Auroc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))
        if _is_multiclass_logits(scores):
            probs = scores.softmax(-1).numpy()
            num_classes = probs.shape[-1]
            return roc_auc_score(
                answers,
                probs,
                average="weighted",
                multi_class="ovr",
                labels=np.arange(num_classes),
            )
        probs = _binary_probs_from_logits(scores)
        return roc_auc_score(answers, probs)

class Auprc(BaseMetric): # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = _binary_probs_from_logits(torch.cat(self.scores))
        answers = _flatten_targets(torch.cat(self.answers))
        return average_precision_score(answers, scores, average='weighted')

class Youdenj(BaseMetric):  # only for binary classification
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = _binary_probs_from_logits(torch.cat(self.scores))
        answers = _flatten_targets(torch.cat(self.answers))
        fpr, tpr, thresholds = roc_curve(answers, scores)
        return float(thresholds[np.argmax(tpr - fpr)])

class F1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))

        if _is_multiclass_logits(scores):  # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = _binary_probs_from_logits(scores)
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)

class Precision(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))

        if _is_multiclass_logits(scores):  # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = _binary_probs_from_logits(scores)
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))

        if _is_multiclass_logits(scores):  # multi-class
            labels = scores.argmax(-1).numpy()
        else:
            scores = _binary_probs_from_logits(scores)
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

class Seqacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self.ignore_indices = (-1, -100)

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        if p.ndim == 0:
            p = p.view(1, 1)
        elif p.ndim == 1:
            p = p.unsqueeze(-1)
        num_classes = p.size(-1)
        self.scores.append(p.reshape(-1, num_classes))
        self.answers.append(t.reshape(-1))

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()
        if scores.size(-1) > 1:
            labels = scores.argmax(-1).numpy()
        else:
            labels = (scores.sigmoid().reshape(-1) >= 0.5).long().numpy()

        # ignore special tokens
        valid_mask = np.ones_like(answers, dtype=bool)
        for idx in self.ignore_indices:
            valid_mask = np.logical_and(valid_mask, answers != idx)
        labels = labels[valid_mask]
        answers = answers[valid_mask]
        if answers.size == 0:
            return 0.0
        return np.nan_to_num(accuracy_score(answers, labels))

class Mse(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores)

class Rmse(Mse):
    def __init__(self):
        super(Rmse, self).__init__()

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_squared_error(answers, scores, squared=False)

class Mae(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_error(answers, scores)

class Mape(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return mean_absolute_percentage_error(answers, scores)

class R2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return r2_score(answers, scores)

class D2(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, *args):
        scores = torch.cat(self.scores).numpy()
        answers = torch.cat(self.answers).numpy()
        return d2_pinball_score(answers, scores)

class Dice(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self, epsilon=1e-6, *args):
        SPATIAL_DIMENSIONS = 2, 3, 4
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers)
        tp = scores.mul(answers).sum(dim=SPATIAL_DIMENSIONS)
        fp = scores.mul(1 - answers).sum(dim=SPATIAL_DIMENSIONS)
        fn = (1 - scores).mul(answers).sum(dim=SPATIAL_DIMENSIONS)
        dice = (tp.mul(2)).div(tp.mul(2).add(fp.add(fn).add(epsilon))).mean()
        return torch.nan_to_num(dice, 0.).item()
        
class Balacc(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = _flatten_targets(torch.cat(self.answers))

        if _is_multiclass_logits(scores):  # multi-class
            labels = scores.argmax(dim=1).numpy()
        else:
            scores = _binary_probs_from_logits(scores)
            if self._use_youdenj: # binary - use Youden's J to determine a label
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores > cutoff, 1, 0)
        return balanced_accuracy_score(answers, labels)


METRIC_REGISTRY = {
    "acc1": Acc1,
    "acc5": Acc5,
    "auroc": Auroc,
    "auprc": Auprc,
    "youdenj": Youdenj,
    "f1": F1,
    "precision": Precision,
    "recall": Recall,
    "seqacc": Seqacc,
    "mse": Mse,
    "rmse": Rmse,
    "mae": Mae,
    "mape": Mape,
    "r2": R2,
    "d2": D2,
    "dice": Dice,
    "balacc": Balacc,
}


def get_metric(metric_name: str):
    key = metric_name.lower()
    if key not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(f"Unknown metric '{metric_name}'. Available: {available}")
    return METRIC_REGISTRY[key]()
