import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from yacs.config import CfgNode
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from src._typing import ArrayLike, CVScheme, Estimator
from src.model import get_model, select_features, get_feature_selector


def train_model(X: ArrayLike, y: ArrayLike, conf: CfgNode) -> ExplainableBoostingClassifier:
    """
    Train an Explainable Boosting Classifier on specified data.

    Parameters
    ----------
    X: ArrayLike of shape (n_obs, n_features)
        A design matrix of feature values.
    y: ArrayLike of shape (n_obs,)
        An array of ground-truths.
    conf: CfgNode
        A yacs configuration node to access configuration values.

    Returns
    -------
    A fitted Explainable Boosting Classifier
    """
    if conf.DATA.SCALE:
        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
        X.loc[:, numeric_cols] = scaler.fit_transform(X.loc[:, numeric_cols])    

    cv = StratifiedKFold(n_splits=conf.EVAL.N_INNER_SPLITS,
                                 shuffle=conf.EVAL.SHUFFLE,
                                 random_state=conf.MISC.SEED)
    feat_subset = select_features(X, y, 
                                  selector=get_feature_selector(conf), 
                                  cv=cv, 
                                  conf=conf)    
                                
    model = get_model(which = conf.TRAIN.WHICH, conf=conf)
    model.fit(X.loc[:, feat_subset], y)

    return model

