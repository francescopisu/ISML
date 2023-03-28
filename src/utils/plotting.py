"""
This file implements the functions needed to produce the figures in the paper.
"""
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from yacs.config import CfgNode


def plot_average_roc_curves(cv_preds: Dict[str, Dict[str, list]],
                            conf: CfgNode,
                            ax: plt.Axes,
                            highlight_best: bool = True) -> plt.Axes:
    """
    Plot average cross-validation ROC curves corresponding to the best models
    found during the inner cross-validation procedure of the nested cv.

    Parameters
    ----------
    cv_preds: Dict[str, Dict[str, list]]
    conf: CfgNode
    ax: plt.Axes
    highlight_best: bool (default = True)

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple of figure and axes.
    """

    ALGO_NAMES_MAP = dict(zip(conf.EVAL.ALGO_SHORT_NAMES, conf.EVAL.ALGO_LONG_NAMES))
    cmap = plt.cm.get_cmap("Set1")
    colors = cmap(np.arange(6))

    # f, ax = plt.subplots(1, 1, figsize=(10, 10))

    for j, (algo_name, outer_cv_data) in enumerate(cv_preds.items()):
        fprs = np.linspace(0, 1, 100)
        tprs = []
        aucs = []

        gts = outer_cv_data['gt']
        probas = outer_cv_data['probas']

        for split_idx, (split_gts, split_probas) in enumerate(zip(gts, probas)):
            auc_val = roc_auc_score(split_gts, split_probas[:, 1])
            aucs.append(auc_val)

            # compute ROC curve components
            fpr, tpr, _ = roc_curve(split_gts, split_probas[:, 1])
            tprs.append(np.interp(fprs, fpr, tpr))

        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std_tprs = tprs.std(axis=0)
        # tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        # tprs_lower = mean_tprs - std_tprs

        fpr, tpr, thresholds = roc_curve(outer_cv_data['gt_conc'], outer_cv_data['probas_conc'][:, 1])

        auc_low, auc_med, auc_up = np.percentile(aucs, [2.5, 50, 97.5])
        auc_cis = "({:.2f} [{:.2f} - {:.2f}])".format(auc_med, auc_low, auc_up)
        lab = "{} {}".format(ALGO_NAMES_MAP[algo_name], auc_cis)

        ax.plot(fpr,
                tpr,
                label=lab,
                lw=5.5 if algo_name == conf.TUNING.ALGORITHM_TO_TUNE and highlight_best else 3.1,
                linestyle=(0, (3, 1, 1, 1)) if algo_name == conf.TUNING.ALGORITHM_TO_TUNE and highlight_best else '-',
                alpha=1.0,
                color=colors[j],
                zorder=100 if algo_name == conf.TUNING.ALGORITHM_TO_TUNE else 1)
        # ax.fill_between(fprs, tprs_lower, tprs_upper,
        #                 color=colors[j],
        #                 alpha=0.08)

    ax.set_xlabel('1 - Specificity',
                  fontdict={"weight": "normal", "size": 16},
                  labelpad=20)
    ax.set_ylabel('Sensitivity',
                  fontdict={"weight": "normal", "size": 16},
                  labelpad=20)
    # plt.title("Algorithm comparison", fontsize=20, pad=20)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray',
            label="Baseline", alpha=.8)
    legend_title = "$\\bf{AUC}$ (95% CI)"
    ax.legend(loc='lower right', fontsize=14, title=legend_title,
              title_fontsize=15, frameon=False)

    return ax


def plot_average_pr_curves(cv_preds: Dict[str, Dict[str, list]],
                           conf: CfgNode,
                           ax: plt.Axes,
                           highlight_best: bool = True,
                           legend_position: str = "lower right") -> plt.Axes:
    """
    Plot average cross-validation precision-recall curves corresponding
    to the best models found during the inner cross-validation procedure of the nested cv.

    Parameters
    ----------
    cv_preds: Dict[str, Dict[str, list]]
    conf: CfgNode
    ax: plt.Axes
    highlight_best: bool (default = True)
    legend_position: str (default = "lower right")

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        A tuple of figure and axes.
    """
    ALGO_NAMES_MAP = dict(zip(conf.EVAL.ALGO_SHORT_NAMES, conf.EVAL.ALGO_LONG_NAMES))
    cmap = plt.cm.get_cmap("Set1")
    colors = cmap(np.arange(6))

    # f, ax = plt.subplots(1, 1, figsize=(10, 10))

    for j, (algo_name, outer_cv_data) in enumerate(cv_preds.items()):
        # fprs = np.linspace(0, 1, 100)
        # tprs = []
        precisions = []
        recalls = np.linspace(0, 1, 100)
        pr_aucs = []

        gts = outer_cv_data['gt']
        probas = outer_cv_data['probas']

        for split_idx, (split_gts, split_probas) in enumerate(zip(gts, probas)):
            pr_auc = average_precision_score(split_gts, split_probas[:, 1])
            pr_aucs.append(pr_auc)

            # compute ROC curve components
            precision, recall, _ = precision_recall_curve(split_gts, split_probas[:, 1])
            precision, recall = precision[::-1], recall[::-1]
            # precisions.append(precision)
            # recalls.append(recall)
            # tprs.append(np.interp(fprs, fpr, tpr))
            prec_array = np.interp(recalls, recall, precision)
            precisions.append(prec_array)

        # tprs = np.array(tprs)
        # mean_tprs = tprs.mean(axis=0)
        # std_tprs = tprs.std(axis=0)
        # tprs_upper = np.minimum(mean_tprs + std_tprs, 1)
        # tprs_lower = mean_tprs - std_tprs
        mean_prec = np.mean(precisions, axis=0)
        std_prec = np.std(precisions, axis=0)

        prec, rec, _ = precision_recall_curve(outer_cv_data['gt_conc'], outer_cv_data['probas_conc'][:, 1])

        pr_auc_low, pr_auc_med, pr_auc_up = np.percentile(pr_aucs, [2.5, 50, 97.5])
        auc_cis = "({:.2f} [{:.2f} - {:.2f}])".format(pr_auc_med, pr_auc_low, pr_auc_up)
        lab = "{} {}".format(ALGO_NAMES_MAP[algo_name], auc_cis)

        # print(len(outer_cv_data['gt_conc']), len(precisions))
        ax.plot(rec,
                prec,
                label=lab,
                lw=5.5 if algo_name == conf.TUNING.ALGORITHM_TO_TUNE and highlight_best else 3.1,
                linestyle=(0, (3, 1, 1, 1)) if algo_name == conf.TUNING.ALGORITHM_TO_TUNE and highlight_best else '-',
                alpha=1.0,
                color=colors[j],
                zorder=100 if algo_name == conf.TUNING.ALGORITHM_TO_TUNE else 1)
        # prec_upper = np.minimum(mean_prec + std_prec, 1)
        # prec_lower = mean_prec - std_prec

        # ax.fill_between(recalls, prec_lower, prec_upper,
        #                 color=colors[j],
        #                 alpha=0.08)

    ax.set_xlabel('Recall',
                  fontdict={"weight": "normal", "size": 16},
                  labelpad=20)
    ax.set_ylabel('Precision',
                  fontdict={"weight": "normal", "size": 16},
                  labelpad=20)
    # plt.title("Algorithm comparison", fontsize=20, pad=20)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))

    # draw baseline
    tot_pos = outer_cv_data['gt_conc'].sum()
    tot = len(outer_cv_data['gt_conc'])
    baseline = tot_pos / tot
    plt.axhline(y=baseline, color='gray', linestyle='--', lw=2, alpha=.8,
                label='Baseline')
    legend_title = "$\\bf{AUC}$ (95% CI)"
    ax.legend(loc=legend_position, fontsize=14, title=legend_title,
              title_fontsize=15, frameon=False)

    return ax
