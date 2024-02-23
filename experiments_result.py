import pickle
from os import PathLike
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

r"""

loss_values list[float], len = num_epochs

                                 p@10,  r@10,  hr@10
epoch_metrics list[ tuple[ dict, float, float, float ] ], len = num_epochs

                          p@10,  r@10,  hr@10
test_metrics tuple[ dict, float, float, float ]

"""


def read_checkpoints_metrics(
    check_point_dir: PathLike  
) -> tuple[ list, list, Any]:
    base_dir = Path(check_point_dir)

    with open(base_dir/"all_loss_values.pkl", "rb") as loss_fp:
        all_loss_values = pickle.load(loss_fp)
    
    with open(base_dir/"all_valid_epoch_metrics.pkl", "rb") as metric_fp:
        all_epoch_metrics = pickle.load(metric_fp)

    with open(base_dir/'test_metrics.pkl', "rb") as test_metrics_dp:
        test_metrics = pickle.load(test_metrics_dp)

    return all_loss_values, all_epoch_metrics, test_metrics


r"""
these figs are needed:

1. compare learning rate loss values
2. compare batch_size validation metrics (recall)
3. compare number of epochs validation metrics (recall)

"""


def print_compared_values(
    xs: list[int],
    ys: list[float],
    fig_path: PathLike,
    y_label: str,
    x_label: str = 'Epochs',
    title: str = 'Recall - HitRate'
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))
    ax[0].plot(xs, ys)
    ax[0].set_title("Loss values")
    ax[0].set_xlabel("HitRate")
    ax[0].set_ylabel("Recall")
    fig.savefig(fig_path)


if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.18))

    svd_hit_rates = [ 0.5312, 0.5153, 0.5494, 0.5877, 0.5884, 0.5933, 0.6108]
    svd_recalls =   [ 0.3306, 0.3200, 0.3321, 0.3736, 0.3724, 0.3869, 0.3875]

    # sort by xs


    kgsvd_hitrates = [0.5301, 0.5595, 0.5670, 0.5705, 0.5793, 0.5895, 0.6876]
    kgsvd_recalls =  [0.4066, 0.4474, 0.4522, 0.4586, 0.4723, 0.4769, 0.5743]

    ax.set_title("HitRate - Recall")
    ax.set_xlabel("HitRate")
    ax.set_ylabel("Recall")
    
    ax.plot(svd_hit_rates, svd_recalls, label="SVD")
    ax.plot(kgsvd_hitrates, kgsvd_recalls, label="KGSVD")

    ax.legend()
    fig.savefig("compare_svd_kgsvd.png")

