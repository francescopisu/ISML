import sys
import numpy as np
import pandas as pd
from typing import Tuple
from yacs.config import CfgNode
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from src._typing import ArrayLike, Estimator, CVScheme

def get_model(which: str, conf: CfgNode = None) -> Estimator: 
    """Returns the specified model.

    Parameters
    ----------
    which : str
        Which model to return. Options are `ebm` for the
        Explainable Boosting Classifier and `lr` for
        classic logistic regression.
    conf : CfgNode
        A yacs configuration node.

    Returns
    -------
    Estimator
        The requested estimator
    """
    if which == "ebm":
        model = ExplainableBoostingClassifier(random_state=conf.MISC.SEED,
                                                interactions=15,
                                                learning_rate=0.02,
                                                min_samples_leaf=5,
                                                outer_bags=35,
                                                inner_bags=35,
                                                max_bins=128,
                                                max_leaves=3,
                                                n_jobs=4)        
    elif which == "lr":
        model = LogisticRegression(penalty="elasticnet", 
                                   solver="saga", 
                                   l1_ratio=0.3,
                                   max_iter=10000,
                                   random_state=conf.MISC.SEED)
    elif which == "svm":
        model = SVC(probability=True, class_weight="balanced", random_state=conf.MISC.SEED)
    elif which == "rf":
        model = RandomForestClassifier(random_state=conf.MISC.SEED,
                                n_estimators=50,
                                max_depth=5,    
                                n_jobs=4)
    elif which == "knn":
        model = KNeighborsClassifier(n_jobs=4)

    return model

def get_feature_selector(conf: CfgNode) -> Estimator:
    fs = RandomForestClassifier(random_state=conf.MISC.SEED,
                                n_estimators=10,
                                max_depth=3,    
                                n_jobs=4,
                                min_samples_leaf=2,
                                min_samples_split=3)
    return fs


def select_features(X: ArrayLike, 
                    y: ArrayLike, 
                    selector: Estimator, 
                    cv: CVScheme, 
                    conf: CfgNode) -> None:
    """Custom feature selection using a feature selector (in our case
    a Random Forest) and cross-validation. We select all features that
    are associated with a Gini impurity reduction at least 25% greater
    than the mean value.

    Parameters
    ----------
    X : ArrayLike
        An array of features values
    y : ArrayLike
        An array of ground truths
    selector : Estimator
        The feature selector. In our case is a Random Forest Classifier
        and we will use the built-in feature importances.
    cv : CVScheme
        A cross-validation scheme for selecting features.
    conf : CfgNode
        A yacs configuration node.

    """
    inner_feature_sets = dict()

    for ji, (train_idxs_inner, _) in enumerate(cv.split(X, y)):
        #print(f"Inner split no. {ji+1}")
        inner_split = f"inner_split_{ji}"
        inner_feature_sets[inner_split] = set()

        # select inner training and validation folds
        X_train_inner = X.iloc[train_idxs_inner, :].copy()
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        y_train_inner = y.iloc[train_idxs_inner]

        #X_val_inner, y_val_inner = X.iloc[val_idxs_inner].copy(), y[val_idxs_inner]

        if conf.DATA.SCALE:
            numeric_cols = X_train_inner.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
            scaler = StandardScaler()
            X_train_inner.loc[:, numeric_cols] = scaler.fit_transform(X_train_inner.loc[:, numeric_cols])
            #X_val_inner.loc[:, numeric_cols] = scaler.transform(X_val_inner.loc[:, numeric_cols])

        # feature selection
        selected_features = set()    
        
        feat_selector = clone(selector)
        feat_selector.fit(X_train_inner, y_train_inner)

        selected_features=list(map(lambda t: t[0], list(filter(lambda w: w[1] > conf.FS.FRAC*feat_selector.feature_importances_.mean(), zip(X.columns, feat_selector.feature_importances_)))))

        if not selected_features:
            continue

        # Check that each output feature is present in the dataset's columns.
        diff = set.difference(set(selected_features), set(X_train_inner.columns))
        if diff:
            raise ValueError(F"Could not find features {DIFF} in the dataframe.")

        if not isinstance(selected_features, set):
            selected_features = set(selected_features)

        inner_feature_sets[inner_split] = selected_features

    return sorted(set.union(*inner_feature_sets.values()))
    

def calibrate_predictions(cal: Tuple[ArrayLike, ArrayLike], 
                          Xtest: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    """Calibrate predicted probabilities using Venn-ABERS method.

    Parameters
    ----------
    cal : ArrayLike
        A tuple of features-ground-truths values to use for calibration.
    Xtest : ArrayLike
        Features values to produce calibrated probabilities of.
    """
    # probability distributions for negative and positive class
    p0, p1 = [], []
    for x in Xtest:
        # add each test point to the calibration data
        # once as negative (label 0)
        cal0 = cal + [(x, 0)]
        iso0 = IsotonicRegression().fit(*zip(*cal0))
        p0.append(iso0.predict([x]))
        
        # once as positive (label 1)
        cal1 = cal + [(x, 1)]
        iso1 = IsotonicRegression().fit(*zip(*cal1))
        p1.append(iso1.predict([x]))

    return np.array(p0).flatten(), np.array(p1).flatten()