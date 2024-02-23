#

import numpy as np

# compute precision@K, recall@K, hit rate@K


def compute_precision(
    pred_item_ids: list[list[int]],
    ground_truth_ids: list[list[int]],
    K,
) -> float:
    """
    Compute precision@K

    Args:
        pred_item_ids: list of list of pred item ids, highest score first, 
            each item_id list for each user
        ground_truth: list of list of true item ids
        K: int
    """
    precision = 0
    for per_pred, per_truth in zip(pred_item_ids, ground_truth_ids):
        per_pred = set(per_pred[:K])
        per_truth = set(per_truth)
        precision += len(per_pred & per_truth) / K
    return precision / len(pred_item_ids)


def compute_recall(
    pred_item_ids: list[list[int]],
    ground_truth_ids: list[list[int]],
    K,
) -> float:
    """
    Compute recall@K

    Args:
        pred_item_ids: list of list of pred item ids, highest score first,
            each item_id list for each user
        ground_truth: list of list of true item ids, user actually interacted items
        K: int
    """
    recall = 0
    for per_pred, per_truth in zip(pred_item_ids, ground_truth_ids):
        per_pred = set(per_pred[:K])
        per_truth = set(per_truth)
        recall += len(per_pred & per_truth) / len(per_truth)
    return recall / len(pred_item_ids)


def compute_hit_rate(
    pred_item_ids: list[list[int]],
    ground_truth_ids: list[list[int]],
    K,
) -> float:
    """
    Compute hit rate@K

    Args:
        pred_item_ids: list of list of pred item ids, highest score first,
            each item_id list for each user
        ground_truth: list of list of true item ids
        K: int
    """
    hit_rate = 0
    for per_pred, per_truth in zip(pred_item_ids, ground_truth_ids):
        per_pred = set(per_pred[:K])
        per_truth = set(per_truth)
        hit_rate += len(per_pred & per_truth) > 0
    return hit_rate / len(pred_item_ids)


def sort_compute(
    pred_scores: np.ndarray,
    truth: np.ndarray,
    K: int
) -> tuple[float, float, float]:
    """
    Shapes:
        pred_scores: (num_preds, 3) - (user_id, item_id, score)
        truth: (num_truths, 3) - (user_id, item_id, weight)
        K: number for @K
    """
    pred_item_ids = []
    truth_ids = []

    users = set(pred_scores[:, 0])
    # which it's a little tricky for this situation is
    # the truth contains not interatced items with the
    # lowest weight, and we need to pick them out.
    for user_id in users:
        # find the record of the user
        pred_user_indices = np.where(pred_scores[:, 0] == user_id)[0]
        truth_user_indices = np.where(truth[:, 0] == user_id)[0]
        user_inters_pred = pred_scores[pred_user_indices]
        user_inter_truths = truth[truth_user_indices]
        # sort the pick content
        pred_args = np.argsort(user_inters_pred[:, 2])[::-1]
        truth_args = np.argsort(user_inter_truths[:, 2])[::-1]
        user_inters_pred = user_inters_pred[pred_args]
        user_inter_truths = user_inter_truths[truth_args]
        # get the actual interacted truth
        lowest_truth_weight = user_inter_truths[-1,2]
        first_not_truth_id = \
            np.greater(user_inter_truths[:, 2], lowest_truth_weight) \
                .astype(np.int32).sum()
        user_interacted_truths = user_inter_truths[:first_not_truth_id]
        if len(user_interacted_truths) == 0:
            print(f"user with user_id {user_id} has no interacted items in valid set, pass")
            continue
        pred_item_ids.append(user_inters_pred[:, 1].astype(np.int32).tolist())
        truth_ids.append(user_interacted_truths[:, 1].astype(np.int32).tolist())

    return (
        compute_precision(pred_item_ids, truth_ids, K),
        compute_recall(pred_item_ids, truth_ids, K),
        compute_hit_rate(pred_item_ids, truth_ids, K)
    )


def compute_metrics_by_answers(
    pred_scores: np.ndarray,
    answers: dict[int, list[int]],
    K: int
) -> tuple[dict[str, list[float]], float, float, float]:
    r"""
    Args:
        pred_scores: pred scores for validation, shape (num_interactions, 3) 
        answers: dict of user_id to actual interacted items
        K: number for @K 
    """
    metrics = dict()
    for user_id, item_ids in answers.items():
        user_indices = np.where(pred_scores[:, 0] == user_id)[0]
        user_scores = pred_scores[user_indices]
        # sort by score
        sorted_indices = np.argsort(user_scores[:, 2])[::-1]
        user_scores = user_scores[sorted_indices]
        
        pred_item_ids = user_scores[:, 1].astype(np.int32).tolist()

        if len(item_ids) == 0:
            print(f"user with user_id {user_id} has no interacted items in valid set, pass")
            continue

        hit_set = set(pred_item_ids[:K]) & set(item_ids)

        metrics.setdefault('precision', []).append(len(hit_set) / K)
        metrics.setdefault('recall', []).append(len(hit_set) / len(item_ids))
        metrics.setdefault('hit_rate', []).append(float(len(hit_set) > 0))
    
    mean_precison = np.mean(metrics['precision'])
    mean_recall = np.mean(metrics['recall'])
    mean_hit_rate = np.mean(metrics['hit_rate'])
    return metrics, mean_precison, mean_recall, mean_hit_rate
