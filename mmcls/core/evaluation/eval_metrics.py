from numbers import Number

import numpy as np
import torch
from sklearn import metrics

def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix
            The shape is (C, C), where C is the number of classes.
    """

    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    assert (
        isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    num_classes = pred.size(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        for t, p in zip(target_label, pred_label):
            confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def precision_recall_f1(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision, recall and f1 score according to the prediction and
    target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        tuple: tuple containing precision, recall, f1 score.

            The type of precision, recall, f1 score is one of the following:

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """

    allowed_average_mode = ['macro', 'none']
    if average_mode not in allowed_average_mode:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    assert (isinstance(pred, np.ndarray) and isinstance(target, np.ndarray)),\
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    if isinstance(thrs, Number):
        thrs = (thrs, )
        return_single = True
    elif isinstance(thrs, tuple):
        return_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    label = np.indices(pred.shape)[1]
    pred_label = np.argsort(pred, axis=1)[:, -1]
    pred_score = np.sort(pred, axis=1)[:, -1]

    precisions = []
    recalls = []
    f1_scores = []
    for thr in thrs:
        # Only prediction values larger than thr are counted as positive
        _pred_label = pred_label.copy()
        if thr is not None:
            _pred_label[pred_score <= thr] = -1
        pred_positive = label == _pred_label.reshape(-1, 1)
        gt_positive = label == target.reshape(-1, 1)
        precision = (pred_positive & gt_positive).sum(0) / np.maximum(
            pred_positive.sum(0), 1) * 100
        recall = (pred_positive & gt_positive).sum(0) / np.maximum(
            gt_positive.sum(0), 1) * 100
        f1_score = 2 * precision * recall / np.maximum(precision + recall,
                                                       1e-20)
        if average_mode == 'macro':
            precision = float(precision.mean())
            recall = float(recall.mean())
            f1_score = float(f1_score.mean())
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    if return_single:
        return precisions[0], recalls[0], f1_scores[0]
    else:
        return precisions, recalls, f1_scores


def precision(pred, target, average_mode='macro', thrs=0.):
    """Calculate precision according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Precision.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    precisions, _, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return precisions


def recall(pred, target, average_mode='macro', thrs=0.):
    """Calculate recall according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: Recall.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, recalls, _ = precision_recall_f1(pred, target, average_mode, thrs)
    return recalls


def f1_score(pred, target, average_mode='macro', thrs=0.):
    """Calculate F1 score according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted mean.
            Defaults to 'macro'.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
         float | np.array | list[float | np.array]: F1 score.

        +----------------------------+--------------------+-------------------+
        | Args                       | ``thrs`` is number | ``thrs`` is tuple |
        +============================+====================+===================+
        | ``average_mode`` = "macro" | float              | list[float]       |
        +----------------------------+--------------------+-------------------+
        | ``average_mode`` = "none"  | np.array           | list[np.array]    |
        +----------------------------+--------------------+-------------------+
    """
    _, _, f1_scores = precision_recall_f1(pred, target, average_mode, thrs)
    return f1_scores


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to the
    prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Support.

            - If the ``average_mode`` is set to macro, the function returns
              a single float.
            - If the ``average_mode`` is set to none, the function returns
              a np.array with shape C.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    with torch.no_grad():
        res = confusion_matrix.sum(1)
        if average_mode == 'macro':
            res = float(res.sum().numpy())
        elif average_mode == 'none':
            res = res.numpy()
        else:
            raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res


def auc(pred_score, target):
    if isinstance(pred_score, torch.Tensor):
        pred_score_np = pred_score.to('cpu').numpy()
    else:
        pred_score_np = pred_score
    # pred_score = np.max(pred_score_np, axis=1)
    # score = pred_score_np[target[:, np.newaxis]]
    # score = []
    # for i in range(pred_score_np.shape[0]):
    #     score.append(pred_score_np[i, target[i]])
    # one_hot = np.zeros(shape=(target.shape[0], target.max() + 1))
    # for i in range(target.shape[0]):
    #     one_hot[i, target[i]] = 1
    # return average_precision_score(one_hot, pred_score)
    fpr, tpr, th = metrics.roc_curve(target, pred_score_np[:, 1])
    return metrics.auc(fpr, tpr)

