import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from yacs.config import CfgNode
from typing import Dict, Callable, List, Tuple, Any
from functools import partial
from sklearn.metrics import (
    precision_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from collections import defaultdict
from interpret.glassbox import ExplainableBoostingClassifier
from tqdm import tqdm

from src._typing import ArrayLike, CVScheme, Estimator
from src.utils.scoring import (
    compute_conf_matrix_metric,
    compute_metrics,
    bootstrap_median_ci
)
from src.utils.io import load_data
from src.model import (
    get_model, 
    select_features, 
    get_feature_selector, 
    calibrate_predictions
)


def compute_cross_val_conf_intervals(cross_val_results: Dict[str, List[float]],
                                     alpha: float = 0.05) -> Dict[str, Tuple[float, float, float]]:
    """
    Computes confidence intervals for the cross-validation results
    with the desired significance level using the percentile method.

    Parameters
    ----------
    cross_val_results: Dict[str, List[float]]
        A dictionary of results of the cross validation procedure.
    alpha: float (default = 0.05)
        The significance level alpha for computing confidence intervals.
        The corresponding confidence level will be (1-alpha)%
        E.g.:
        alpha = 0.05
        confidence level = (1-0.05)% = 95%

    Returns
    -------
    Dict[str, Tuple[float, float, float]]
        A dictionary where keys are of the form SET_METRIC and values are
        tuple of median score, lower and upper bound of confidence interval.
    """
    results_ = dict()
    for k, scores in cross_val_results.items():
        med = np.median(scores).round(3)
        ci_lower = np.percentile(scores, alpha * 100 / 2).round(2)
        ci_upper = np.percentile(scores, 100 - (alpha * 100 / 2)).round(2)

        results_[k] = (med, ci_lower, ci_upper)

    return results_


def test_performance(conf: CfgNode,
                     model: Estimator,
                     X_test: ArrayLike, y_test: ArrayLike,
                     eval_metrics: Dict[str, Callable]) -> Dict[str, str]:
    """
    Compute test set performance of the specified model.
    In greater detail, the model is trained on the whole training set
    and used to predict probabilities (if is suppoted by the model)
    on the test data. Then, median and 95% CI are computed for each
    evaluation metric and returned in a dictionary.

    Parameters
    ----------
    conf: CfgNode
        A yacs configuration node to access configuration values.
    model : Estimator
      A fitted estimator to be tested on test data.
    X_test: ArrayLike of shape (n_obs, n_features)
        Design matrix containing feature values of test data.
    y_test: ArrayLike of shape (n_obs,)
        A vector of test data ground-truth labels.
    eval_metrics: Dict[str, Callable]
        A dictionary of evaluation metrics.
        Example:
        {
            "auc": roc_auc_score,
            "brier_loss": brier_score_loss,
            ...
        }

    Returns
    -------
    Dict[str, str]
        A dictionary of metrics values computed on the test set.
    """
    # fit scaler on train data and apply scaling to test data
    # scaler = StandardScaler()
    # X_train, y_train = load_data(conf, which='train')
    # numeric_cols = X_train.select_dtypes(include=np.float64).columns.tolist()
    # scaler.fit(X_train[numeric_cols])
    # X_test.loc[:, numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Predict on test set
    pred_probas_test = model.predict_proba(X_test)
    pos_class_probas = pred_probas_test[:, 1]

    # predicted labels with class. threshold of 0.5
    pred_test_labels = np.where(pos_class_probas > conf.EVAL.THRESHOLD, 1, 0)

    out = {}
    for metric_name, metric_fn in eval_metrics.items():
        print("Bootstrapping {}..".format(metric_name))
        if metric_name in ['ROCAUC', 'PRAUC']:
            # use probabilities
            preds = pos_class_probas
        else:
            # use labels
            preds = pred_test_labels

        med, conf_int = bootstrap_median_ci(target=y_test,
                                            preds=preds,
                                            metric=metric_fn,
                                            n_boot=conf.EVAL.BOOT_ROUNDS,
                                            seed=conf.MISC.SEED)
        print(f"Metric name: {metric_name}, Median value:{med} ")
        out[metric_name] = "{:.2f} [{:.2f}-{:.2f}]".format(med, conf_int[0], conf_int[1])

    return out

def cross_validate(X: ArrayLike,
                   y: ArrayLike,
                   outer_cv: CVScheme,
                   inner_cv: CVScheme,
                   conf: CfgNode) -> List[Tuple[Estimator, Dict, Dict]]:
    """
    Estimate generalization performance through cross-validation.

    Parameters
    ----------
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    outer_cv: CVScheme
        A cross-validation scheme for model evaluation.
    inner_cv: CVScheme
        A cross-validation scheme for feature selection.        
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple consisting of average values of evaluation metrics and
        predicted probabilities for each subject (out-of-fold).
    """
    pd.options.mode.chained_assignment = None
    results = defaultdict(list)
    preds = dict()

    eval_metrics = get_evaluation_metrics()

    preds = dict()
    gts = []
    gts_train = []
    gts_idxs = []
    cv_val_idxs = []
    probas = []
    probas_train = []
    probas_low = []
    probas_high = []
    probas_low_train = []
    probas_high_train = []
    errors = []
    correct = []
    models = []

    features = dict()

    rng = np.random.default_rng(conf.MISC.SEED)
    rand_nums = rng.choice(5000, size=conf.EVAL.N_REPEATS, replace = False)

    # repeat the whole cross-validation analysis N_REPEATS times
    for it in tqdm(range(conf.EVAL.N_REPEATS), total=conf.EVAL.N_REPEATS):
        X_sh, y_sh = shuffle(X, y, random_state=rand_nums[it])        
        
        it_key = f"it_{it}"
        if it_key not in features:
            features[it_key] = dict()

        for i, (train_idx, test_idx) in tqdm(enumerate(outer_cv.split(X_sh, y_sh)), total=outer_cv.get_n_splits(X_sh, y_sh), leave=False):
            Xtemp, ytemp = X_sh.iloc[train_idx, :], y_sh.iloc[train_idx]
            Xtest, ytest = X_sh.iloc[test_idx, :], y_sh.iloc[test_idx]
            
            cv_val_idxs.append(test_idx)
            gts_idxs.append(ytest.index)

            cv_key = f"outer_split_{i}"

            if conf.EVAL.CALIBRATE:
                # split Xtemp, ytemp into Xtrain, ytrain and Xcal, ycal (15%)
                Xtrain, Xcal, ytrain, ycal = train_test_split(Xtemp, ytemp, 
                                        test_size=conf.EVAL.CALIB_FRAC, 
                                        stratify=ytemp, 
                                        random_state=conf.MISC.SEED)
                #print(Xtrain.shape, Xcal.shape)
            else:
                Xtrain, ytrain = Xtemp, ytemp
            
            # instantiate EBM
            model = get_model(which=conf.TRAIN.WHICH, conf=conf)

            # select features
            feat_subset = select_features(Xtrain, ytrain, 
                                        selector=get_feature_selector(conf), 
                                        cv=inner_cv, 
                                        conf=conf)
            features[it_key][cv_key] = feat_subset

            # scale numeric variables
            if conf.DATA.SCALE:
                numeric_cols = Xtrain.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
                scaler = StandardScaler()
                Xtrain.loc[:, numeric_cols] = scaler.fit_transform(Xtrain.loc[:, numeric_cols])
                Xtest.loc[:, numeric_cols] = scaler.transform(Xtest.loc[:, numeric_cols])


            # fit model using the reduced feature set
            model.fit(Xtrain.loc[:, feat_subset], ytrain)

            if conf.EVAL.CALIBRATE:
                calib_model = CalibratedClassifierCV(estimator=model,
                                                     method="sigmoid",
                                                     n_jobs=10,
                                                     cv="prefit")
                calib_model.fit(Xcal.loc[:, feat_subset], ycal)
            else:
                calib_model = model

            models.append(calib_model)

            test_preds = calib_model.predict_proba(Xtest.loc[:, feat_subset])
            train_preds = calib_model.predict_proba(Xtrain.loc[:, feat_subset])

            # pool ground-truths and predicted probabilities
            # on both train and validation subsets
            gts.append(ytest)
            gts_train.append(ytrain)
            probas.append(test_preds)
            probas_train.append(train_preds)

            # compute youden's index on training portion
            fpr, tpr, thresholds = roc_curve(ytrain, train_preds[:, 1])
            idx = np.argmax(tpr - fpr)
            youden = thresholds[idx]

            # use threshold to classify validation instances
            labels_val = np.where(test_preds[:, 1] >= youden, 1, 0)

            # find out errors
            error_idxs = Xtest[(ytest != labels_val)].index
            errors.append(error_idxs)
            correct.append(Xtest[(ytest == labels_val)].index)

            # compute evaluation metrics
            test_scores = compute_metrics(test_preds, ytest, eval_metrics, threshold=youden)
            train_scores = compute_metrics(train_preds, ytrain, eval_metrics, threshold=youden)

            # save scores
            for s, scores_dict in zip(conf.EVAL.SET_NAMES, [train_scores, test_scores]):
                for metric_name, score in scores_dict.items():
                    key = "{}_{}_{}".format("EBM", s, metric_name)
                    results[key].append(score)

    # concatenate ground-truths and predicted probabilities
    preds["EBM"] = {
        "gt_conc": np.concatenate(gts),
        "gt_conc_train": np.concatenate(gts_train),
        "probas_conc": np.concatenate(probas),
        "probas_conc_train": np.concatenate(probas_train),
        "gt": gts,
        "gt_train": gts_train,
        "gt_idxs": gts_idxs,
        "cv_val_idxs": cv_val_idxs, # to be used on shuffled dataset
        "probas": probas,
        "probas_train": probas_train,
        "probas_low": probas_low,
        "probas_high": probas_high,
        "probas_low_train": probas_low_train,
        "probas_high_train": probas_high_train,
        "errors": errors,
        "correct": correct,
        "features": features,
        "models": models
    }

    # compute confidence intervals based on the percentile method.
    results_with_cis = compute_cross_val_conf_intervals(results, alpha=conf.EVAL.ALPHA)

    return results_with_cis, preds


def get_evaluation_metrics() -> Dict[str, Callable]:
    eval_metrics = {
        "Sensitivity": partial(compute_conf_matrix_metric, metric_name="tpr"),
        "Specificity": partial(compute_conf_matrix_metric, metric_name="tnr"),
        "PPV": partial(compute_conf_matrix_metric, metric_name="ppv"),
        "NPV": partial(compute_conf_matrix_metric, metric_name="npv"),
        "ROCAUC": roc_auc_score,
        "PRAUC": average_precision_score,
        "Brier": brier_score_loss
    }

    return eval_metrics
